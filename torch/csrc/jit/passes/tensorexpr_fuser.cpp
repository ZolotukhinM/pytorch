#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <ATen/record_function.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/passes/constant_pooling.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {
void CreateFunctionalGraphs2(const std::shared_ptr<Graph>& graph);

namespace tensorexpr {
bool isSupported(Node* node) {
  // TODO:
  switch (node->kind()) {
    case aten::add:
    case aten::_cast_Float:
    case aten::type_as:
    case aten::sub:
    case aten::mul:
    case aten::div:
    case aten::eq:
    case aten::ne:
    case aten::ge:
    case aten::gt:
    case aten::le:
    case aten::lt:
    case aten::pow:
    case aten::clamp:
    case aten::lerp:
    case aten::log10:
    case aten::log:
    case aten::log2:
    case aten::exp:
    case aten::erf:
    case aten::erfc:
    case aten::fmod:
    case aten::cos:
    case aten::sin:
    case aten::tan:
    case aten::acos:
    case aten::asin:
    case aten::atan:
    case aten::atan2:
    case aten::cosh:
    case aten::sinh:
    case aten::tanh:
    case aten::sqrt:
    case aten::rsqrt:
    case aten::abs:
    case aten::floor:
    case aten::ceil:
    case aten::round:
    case aten::trunc:
    case aten::threshold:
    case aten::remainder:
    case prim::ConstantChunk:
    case aten::cat:
    case prim::ListConstruct:
    case aten::sigmoid:
    case aten::relu:
    case aten::addcmul:
    case aten::neg:
    case aten::reciprocal:
    case aten::expm1:
    case aten::lgamma:
    case aten::slice:
    case aten::unsqueeze:
    case aten::frac:
    // TODO: uncomment once we can handle rand+broadcasts
    // case aten::rand_like:
    case aten::_sigmoid_backward:
    case aten::_tanh_backward:
    case aten::__and__:
    case aten::__or__:
    case aten::__xor__:
    case aten::__lshift__:
    case aten::__rshift__:
    case aten::where:
      return true;
    // Operators that can be both elementwise or reductions:
    case aten::min:
    case aten::max:
      if (node->inputs().size() != 2) {
        return false;
      }
      if (!node->inputs()[0]->type()->cast<TensorType>() ||
          !node->inputs()[1]->type()->cast<TensorType>()) {
        return false;
      }
      return true;
    default:
      return false;
  }
}
} // namespace tensorexpr

static bool texpr_fuser_enabled_ = false;
void setTensorExprFuserEnabled(bool val) {
  texpr_fuser_enabled_ = val;
}

bool tensorExprFuserEnabled() {
  static const char* enable_c_str = std::getenv("PYTORCH_TENSOREXPR");
  if (!enable_c_str) {
    return texpr_fuser_enabled_;
  }
  if (std::string(enable_c_str) == "0") {
    return false;
  }
  return true;
}

const Symbol& getTensorExprSymbol() {
  static Symbol s = Symbol::fromQualString("tensorexpr::Group");
  return s;
}

value_list sortReverseTopological(
    ArrayRef<torch::jit::Value*> inputs,
    torch::jit::Block* block) {
  value_list result;
  for (auto i : inputs) {
    if (i->node()->owningBlock() == block) {
      result.push_back(i);
    }
  }
  // Sort in reverse topological order
  std::sort(
      result.begin(),
      result.end(),
      [&](torch::jit::Value* a, torch::jit::Value* b) {
        return a->node()->isAfter(b->node());
      });
  return result;
}

bool allShapesAreKnown(Value* v) {
  if (!v->type()->cast<TensorType>()) {
    return true;
  }
  return v->isCompleteTensor();
}

bool allShapesAreKnown(Node* node) {
  // TODO: Relax the checks to support dynamic shapes
  for (torch::jit::Value* output : node->outputs()) {
    if (!allShapesAreKnown(output)) {
      return false;
    }
  }
  for (torch::jit::Value* input : node->inputs()) {
    if (!allShapesAreKnown(input)) {
      return false;
    }
  }
  return true;
}

bool canHandle(Node* node) {
  if (node->kind() == prim::Constant) {
    if (node->output()->type()->cast<TensorType>()) {
      // TODO: add support for tensor constants.
      return false;
    }
    return true;
  }
  if (node->kind() == prim::profile) {
    return true;
  }
  if (node->kind() == prim::Loop) {
    return false; // TODO
  }
//   if (!allShapesAreKnown(node)) {
//     return false;
//   }

  // Don't include nodes whose inputs are tensor constants - we cannot handle
  // them at the moment.
  // TODO: actually support tensor constants and remove this.
  for (torch::jit::Value* input : node->inputs()) {
    if (input->node()->kind() == prim::Constant &&
        input->type()->cast<TensorType>()) {
      return false;
    }
  }
  return tensorexpr::isSupported(node);
}

bool canHandle(Node* node, AliasDb& aliasDb) {
  return canHandle(node);
}


#define REQ(cond)                           \
  if (!(cond)) {                            \
    GRAPH_DEBUG("Failed cond " #cond "\n"); \
    return false;                           \
  }

bool canMerge(Node* consumer, Node* producer, AliasDb& aliasDb, std::unordered_map<Value*, TensorTypePtr> value_types) {
  // Only handle complete tensor types
  for (torch::jit::Value* output : consumer->outputs()) {
    REQ(output->isCompleteTensor() ||
        (value_types.count(output) && value_types.at(output)->isComplete()));
  }

  // Only fuse within a block
  REQ(consumer->owningBlock() == producer->owningBlock());

  // Symbolic checks
  REQ(canHandle(producer, aliasDb));
  REQ(
      (canHandle(consumer, aliasDb) ||
       consumer->kind() == getTensorExprSymbol()));

  // Alias checks
  REQ(aliasDb.couldMoveBeforeTopologically(producer, consumer));

  // Ops that return aliases can only be folded if this is the only use.
  if (producer->kind() == aten::slice || producer->kind() == aten::unsqueeze ||
      producer->kind() == prim::ConstantChunk) {
    for (auto& use : producer->output(0)->uses()) {
      REQ(use.user == consumer);
    }
  }

  if (!consumer->hasAttribute(attr::Subgraph) &&
      consumer->kind() != getTensorExprSymbol()) {
    // Don't initiate a fusion group from prim::ListConstruct
    REQ(consumer->kind() != prim::ListConstruct);
    REQ(consumer->kind() != aten::slice);
    REQ(consumer->kind() != aten::unsqueeze);
    REQ(consumer->kind() != prim::ConstantChunk);

    // Don't initiate a fusion group just for a constant operand
    REQ(producer->kind() != prim::Constant);
  }

  if (producer->kind() == aten::cat) {
    REQ(producer->inputs()[0]->node()->kind() == prim::ListConstruct);
    REQ(producer->inputs()[0]->uses().size() == 1);
    REQ(producer->inputs()[1]->node()->kind() == prim::Constant);
  } else if (consumer->kind() == aten::cat) {
    REQ(consumer->inputs()[0]->node()->kind() == prim::ListConstruct);
    REQ(consumer->inputs()[0]->uses().size() == 1);
    REQ(consumer->inputs()[1]->node()->kind() == prim::Constant);
  }

  return true;
}
#undef REQ

Node* getOrCreateTensorExprSubgraph(Node* n) {
  if (n->hasAttribute(attr::Subgraph) && n->kind() == getTensorExprSymbol()) {
    return n;
  }
  auto te_group =
      SubgraphUtils::createSingletonSubgraph(n, getTensorExprSymbol());
  GRAPH_UPDATE("getOrCreateTensorExprSubgraph: ", *te_group);
  return te_group;
}

c10::optional<Node*> tryMerge(
    Node* consumer,
    Node* producer,
    AliasDb& aliasDb, std::unordered_map<Value*, TensorTypePtr> value_types) {
  GRAPH_DEBUG(
      "Trying producer ",
      getHeader(producer),
      " and consumer ",
      getHeader(consumer),
      ":\n");

  if (!canMerge(consumer, producer, aliasDb, value_types)) {
    return c10::nullopt;
  }

  consumer = getOrCreateTensorExprSubgraph(consumer);

  if (producer->kind() == aten::cat) {
    Node* listconstruct = producer->inputs()[0]->node();

    aliasDb.moveBeforeTopologicallyValid(producer, consumer);
    GRAPH_UPDATE(
        "Merging ", getHeader(producer), " into ", getHeader(consumer));
    SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);

    aliasDb.moveBeforeTopologicallyValid(listconstruct, consumer);
    GRAPH_UPDATE(
        "Merging ", getHeader(listconstruct), " into ", getHeader(consumer));
    SubgraphUtils::mergeNodeIntoSubgraph(listconstruct, consumer);
  } else {
    aliasDb.moveBeforeTopologicallyValid(producer, consumer);
    GRAPH_UPDATE(
        "Merging ", getHeader(producer), " into ", getHeader(consumer));
    SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);
  }

  return consumer;
}

std::pair<graph_node_list::iterator, bool> scanNode(
    Node* consumer,
    AliasDb& aliasDb, std::unordered_map<Value*, TensorTypePtr> value_types) {
  auto inputs =
      sortReverseTopological(consumer->inputs(), consumer->owningBlock());

  // Grab the iterator below consumer.  We'll use that to determine
  // where to resume iteration, even if consumer gets relocated within
  // the block.
  auto iter = --consumer->reverseIterator();
  for (auto input : inputs) {
    if (auto group = tryMerge(consumer, input->node(), aliasDb, value_types)) {
      // Resume iteration from where consumer is/used to be.
      return {++iter, true};
    }
  }

  // We know consumer didn't move, so skip over it.
  return {++(++iter), false};
}


void removeProfilingNodes_(Block* b) {
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    if (it->kind() == prim::profile) {
      if (it->outputs().size()) {
//         it->input()->setType(it->output()->type());
        it->output()->replaceAllUsesWith(it->input());
      }
      it.destroyCurrent();
    } else {
      for (Block* ib : it->blocks()) {
        removeProfilingNodes_(ib);
      }
    }
  }
}

void findValuesWithKnownSizes_(
    Block* block,
    std::unordered_map<Value*, TensorTypePtr>& value_types) {
  auto reverse_iter = block->nodes().reverse();
  for (auto it = reverse_iter.begin(); it != reverse_iter.end();) {
    Node* n = *it++;

    for (torch::jit::Value* output : n->outputs()) {
      if (allShapesAreKnown(output)) {
        if (auto tensor_ty = output->type()->cast<TensorType>()) {
          value_types[output] = tensor_ty;
        }
      }
    }

    // constants get copied into the graph
    if (n->kind() == prim::profile && n->outputs().size() == 1 &&
        allShapesAreKnown(n->output())) {
      if (auto tensor_ty = n->output()->type()->cast<TensorType>()) {
        std::cerr << "Known type for: %" << n->input()->debugName() << "; type: " << *tensor_ty << "\n";
        value_types[n->input()] = tensor_ty;
      }
    }

    for (Block* b : n->blocks()) {
      findValuesWithKnownSizes_(b, value_types);
    }
  }
}

void insertGuards(
    Block* b,
    std::unordered_map<Value*, TensorTypePtr> value_types) {
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    if (it->kind() == getTensorExprSymbol()) {
      // Fixup types in subgraph inputs
      auto te_g = it->g(attr::Subgraph);
      for (size_t idx = 0; idx < te_g->inputs().size(); idx++) {
        auto inp = it->input(idx);
        if (value_types.count(inp) && value_types.at(inp)->isComplete()) {
          te_g->inputs().at(idx)->setType(value_types.at(inp));
        }
      }

      // Add guard for inputs
      for (auto inp : it->inputs()) {
        if (value_types.count(inp) && value_types.at(inp)->isComplete()) {
          auto guard = b->owningGraph()->create(prim::Guard, {inp}, 1);
          auto go = guard->output();
          go->setType(value_types.at(inp));
          guard->insertBefore(*it);
          inp->replaceAllUsesWith(go);
          guard->replaceInput(0, inp);
        }
      }

    } else {
      for (Block* ib : it->blocks()) {
        insertGuards(ib, value_types);
      }
    }
  }
}

void FuseTensorExprs(std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("Before TExprFuser: ", graph);

  std::unordered_map<Value*, TensorTypePtr> value_types;

  findValuesWithKnownSizes_(graph->block(), value_types);
  removeProfilingNodes_(graph->block());

  // Get rid of dead code so that we don't waste effort fusing it.
  EliminateDeadCode(graph);

  AliasDb aliasDb(graph);
  auto block = graph->block();
//   CreateFunctionalGraphs2(graph);

  std::vector<std::pair<graph_node_list_iterator, graph_node_list_iterator>>
      worklist;
  std::unordered_set<torch::jit::Block*> visited_blocks;

  bool any_changed = true;
  while (any_changed) {
    any_changed = false;
    worklist.push_back({block->nodes().rbegin(), block->nodes().rend()});

    while (worklist.size()) {
      auto& it = worklist.back().first;
      auto end = worklist.back().second;

      if (it->blocks().size()) {
        Node* n = *it;
        ++it;

        if (it == end) {
          worklist.pop_back();
        }

        for (auto b : n->blocks()) {
          if (!visited_blocks.count(b)) {
            worklist.push_back({b->nodes().rbegin(), b->nodes().rend()});
            visited_blocks.insert(b);
          }
        }
      } else {
        bool changed;
        std::tie(it, changed) = scanNode(*it, aliasDb, value_types);
        any_changed |= changed;
        if (it == end) {
          worklist.pop_back();
        }
      }
    }
  }
  insertGuards(graph->block(), value_types);

  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);

  GRAPH_DUMP("After TExprFuser: ", graph);
}


/////////////////////////////////////////////////////////////////////////


struct FunctionalGraphSlicer2 {
  FunctionalGraphSlicer2(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  void run() {
    findValuesWithKnownSizes(graph_->block());
//     for (auto kv : value_types_) {
//       std::cerr << "Value: %" << kv.first->debugName() << ", type: " << *kv.second << "\n";
//     }

    removeProfilingNodes(graph_->block());
//     return;
    bool changed = true;
    // TODO: more sane strategy
    size_t MAX_NUM_ITERATIONS = 4;

    // First, analyze the functional subset of the graph, and then create
    // functional graphs. The graph gets mutated when we create functional
    // subgraphs, invalidating the AliasDb, so we need to do our analysis
    // first.
    for (size_t i = 0; i < MAX_NUM_ITERATIONS && changed; ++i) {
      aliasDb_ = torch::make_unique<AliasDb>(graph_);
      AnalyzeFunctionalSubset(graph_->block());
      changed = CreateFunctionalGraphsImpl(graph_->block());
    }
  }

 private:
  std::unordered_set<Value*> values_with_known_sizes_;
  std::unordered_map<Value*,TensorTypePtr> value_types_;
  bool isEmptyFunctionalGraph(Node* n) {
    auto g = n->g(attr::Subgraph);
    return g->inputs().size() == 0 && g->outputs().size() == 0;
  }

  void nonConstNodes(Block* block, size_t* num) {
    for (auto it = block->nodes().begin();
         it != block->nodes().end() && *num < minSubgraphSize_;
         ++it) {
      Node* n = *it;
      if (n->kind() == prim::Constant) {
        continue;
      }
      *num = *num + 1;
      for (Block* b : n->blocks()) {
        nonConstNodes(b, num);
      }
    }
  }

  bool inlineIfTooSmall(Node* n) {
    AT_ASSERT(n->kind() == getTensorExprSymbol());
    auto subgraph = SubgraphUtils::getSubgraph(n);
    size_t num_modes = 0;
    nonConstNodes(subgraph->block(), &num_modes);
    if (num_modes < minSubgraphSize_) {
      SubgraphUtils::unmergeSubgraph(n);
      return true;
    }
    return false;
  }

  bool isFusible(Node* node) {
    if (node->kind() == prim::Constant) {
      if (node->output()->type()->cast<TensorType>()) {
        // TODO: add support for tensor constants.
        return false;
      }
      return true;
    }
    if (node->kind() == prim::profile) {
      return true;
    }
    if (node->kind() == prim::Loop) {
      return false; // TODO
    }
    for (torch::jit::Value* input : node->inputs()) {
      if (!values_with_known_sizes_.count(input)) {
        return false;
      }
    }

    // Don't include nodes whose inputs are tensor constants - we cannot handle
    // them at the moment.
    // TODO: actually support tensor constants and remove this.
    for (torch::jit::Value* input : node->inputs()) {
      if (input->node()->kind() == prim::Constant &&
          input->type()->cast<TensorType>()) {
        return false;
      }
    }
    return tensorexpr::isSupported(node);
  }

  void removeProfilingNodes(Block* b) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
      if (it->kind() == prim::profile) {
        if (it->outputs().size()) {
          it->output()->replaceAllUsesWith(it->input());
        }
        it.destroyCurrent();
      } else {
        for (Block* ib : it->blocks()) {
          removeProfilingNodes(ib);
        }
      }
    }
  }

  void findValuesWithKnownSizes(Block* block) {
    auto reverse_iter = block->nodes().reverse();
    for (auto it = reverse_iter.begin(); it != reverse_iter.end();) {
      Node* n = *it++;

      for (torch::jit::Value* output : n->outputs()) {
        if (allShapesAreKnown(output)) {
          if (auto tensor_ty = output->type()->cast<TensorType>()) {
            value_types_[output] = tensor_ty;
//           std::cerr << "SET Value: %" << output->debugName() << ", type: " << *tensor_ty << "\n";
          }
          values_with_known_sizes_.insert(output);
        }
      }

      // constants get copied into the graph
      if (n->kind() == prim::profile && n->outputs().size() == 1 && allShapesAreKnown(n->output())) {
        if (auto tensor_ty = n->output()->type()->cast<TensorType>()) {
//           std::cerr << "SET Value: %" << n->input()->debugName() << ", type: " << *tensor_ty << "\n";
          value_types_[n->input()] = tensor_ty;
        }
        values_with_known_sizes_.insert(n->input());
      }


      for (Block* b : n->blocks()) {
        findValuesWithKnownSizes(b);
      }
    }
  }

  bool CreateFunctionalGraphsImpl(Block* block) {
    /*
    Iterate the block in reverse and create FunctionalSubgraphs.
    When we encounter a node that isn't functional, we skip it. Otherwise,
    we try to merge the functional node into the current functional subgraph.
    If it can't be merged into the current functional subgraph node, then we
    start a functional subgraph group.
    */
    bool changed = false;
    std::vector<Node*> functional_graph_nodes;

    Node* functional_subgraph_node =
        graph_->createWithSubgraph(getTensorExprSymbol())
            ->insertBefore(block->return_node());
    auto reverse_iter = block->nodes().reverse();
    std::vector<Value*> graph_outputs;
    for (auto it = reverse_iter.begin(); it != reverse_iter.end();) {
      Node* n = *it++;

      // constants get copied into the graph
      if (n->kind() == prim::Constant || n == functional_subgraph_node) {
        continue;
      }

      // if `n` is functional, all of its blocks will be merged into the
      // new functional subgraph, so we only need to recurse if it is not
      // functional
      if (!functional_nodes_.count(n) || !isFusible(n)) {
        for (Block* b : n->blocks()) {
          auto block_changed = CreateFunctionalGraphsImpl(b);
          changed = block_changed && changed;
        }
        continue;
      }

      if (n->kind() == getTensorExprSymbol() &&
          isEmptyFunctionalGraph(functional_subgraph_node)) {
        functional_subgraph_node->destroy();
        functional_subgraph_node = n;
        continue;
      }

      changed = true;
      if (aliasDb_->moveBeforeTopologicallyValid(n, functional_subgraph_node)) {
        SubgraphUtils::mergeNodeIntoSubgraph(n, functional_subgraph_node);
      } else {
        functional_graph_nodes.emplace_back(functional_subgraph_node);
        functional_subgraph_node =
            graph_->createWithSubgraph(getTensorExprSymbol())->insertAfter(n);
        SubgraphUtils::mergeNodeIntoSubgraph(n, functional_subgraph_node);
      }
    }
    functional_graph_nodes.emplace_back(functional_subgraph_node);
    for (Node* functional_node : functional_graph_nodes) {
      if (!inlineIfTooSmall(functional_node)) {
        ConstantPooling(functional_node->g(attr::Subgraph));
      }
      for (size_t i = 0; i < functional_node->inputs().size(); i++) {
        if (!value_types_.count(functional_node->input(i))) {
          continue;
        }
        // TODO: insert guards
        functional_node->g(attr::Subgraph)->inputs()[i]->setType(value_types_.at(functional_node->input(i)));
      }
    }
    return changed;
  }

  bool AnalyzeFunctionalSubset(Node* n) {
    // TODO: clarify hasSideEffects, isNondeterministic
    bool is_functional_node = true;

    // Functional Graphs are not responsible for maintaining aliasing
    // relationships. If an output of a functional graph escapes scope
    // or is mutated then we might change semantics of the program if
    // aliasing relationships are changed.
    // We don't allow any node in the functional graph to output a value
    // that escapes scope or is mutated, and we don't allow any mutating nodes
    // into the graph.
    // - allow functional graphs to have at most one value that can escape scope
    // - allow outputs which alias the wildcard set but do not "re-escape"
    for (Value* v : n->outputs()) {
      bool has_writers = aliasDb_->hasWriters(v);
      bool escapes_scope = aliasDb_->escapesScope(v);
      if (has_writers) {
        mutated_values_.insert(v);
      }
      is_functional_node = is_functional_node && !escapes_scope && !has_writers;
    }

    for (Block* block : n->blocks()) {
      auto functional_block = AnalyzeFunctionalSubset(block);
      is_functional_node = is_functional_node && functional_block;
    }

    is_functional_node = is_functional_node && !aliasDb_->isMutable(n);
    if (is_functional_node) {
      functional_nodes_.insert(n);
    }
    return is_functional_node;
  }

  void AnalyzeFunctionalSubset(at::ArrayRef<Block*> blocks) {
    for (Block* block : blocks) {
      AnalyzeFunctionalSubset(block);
    }
  }

  bool AnalyzeFunctionalSubset(Block* block) {
    bool is_functional_block = true;
    // block inputs will not yet have been iterated through,
    // so we need to add them to our set of mutated & escape values.
    for (Value* v : block->inputs()) {
      bool has_writers = aliasDb_->hasWriters(v);
      if (has_writers) {
        mutated_values_.insert(v);
      }
    }
    // if a block output is not functional, then the corresponding output for
    // the node that contains the block will not be functional either, so we do
    // not need to analyze the block outputs here.
    for (Node* n : block->nodes()) {
      bool functional = AnalyzeFunctionalSubset(n);
      is_functional_block = is_functional_block && functional;
    }
    return is_functional_block;
  }

  std::unordered_set<Node*> functional_nodes_;
  std::unordered_set<Value*> mutated_values_;
  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  size_t minSubgraphSize_ = 2;
};

void InlineFunctionalGraphs(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++;
    for (Block* b : n->blocks()) {
      InlineFunctionalGraphs(b);
    }
    if (n->kind() == getTensorExprSymbol()) {
      EliminateCommonSubexpression(n->g(attr::Subgraph));
      SubgraphUtils::unmergeSubgraph(n);
    }
  }
}

struct MutationRemover {
  MutationRemover(const std::shared_ptr<Graph>& graph)
      : aliasDb_(nullptr), graph_(graph) {
    aliasDb_ = torch::make_unique<AliasDb>(graph_);
  }

  void removeListMutation() {
    RemoveListMutation(graph_->block());
  }

  void removeTensorMutation() {
    RemoveTensorMutation(graph_->block());
  }

 private:
  bool newMemoryLocation(Value* v) {
    // bail on nodes with side effects, blocks, or graph / graph inputs
    Node* n = v->node();
    bool unhandled_node = n->blocks().size() != 0 ||
        n->hasAttribute(attr::Subgraph) || n->hasSideEffects() ||
        (v->node()->kind() == prim::Param);

    // if the output isn't contained or alias by the inputs to its node, it's
    // unique
    return !unhandled_node &&
        !aliasDb_->mayContainAlias(v->node()->inputs(), v) &&
        !(v->node()->kind() == prim::Param);
  }

  bool inplaceOpVariant(Node* n) {
    if (!n->kind().is_aten()) {
      return false;
    }
    auto name = n->schema().name();
    bool inplace_op = name.at(name.size() - 1) == '_';
    if (!inplace_op) {
      return false;
    }

    // needs to have alias analysis by schema
    auto op = n->maybeOperator();
    if (!op) {
      return false;
    }
    if (op->aliasAnalysisKind() != AliasAnalysisKind::FROM_SCHEMA) {
      return false;
    }

    // all inplace ops at time of writing have a single input that is mutated
    // and returned. check that this is true, anything else could have strange
    // semantics,
    if (n->outputs().size() != 1 || n->inputs().size() == 0) {
      return false;
    }
    auto inputs = n->inputs();
    if (!aliasDb_->writesToAlias(n, {inputs.at(0)}) ||
        aliasDb_->writesToAlias(
            n, {inputs.slice(1).begin(), inputs.slice(1).end()})) {
      return false;
    }

    auto new_schema = name.substr(0, name.size() - 1);
    return getAllOperatorsFor(Symbol::fromQualString(new_schema)).size() != 0;
  }

  bool listAppendFollowingListConstruct(Node* n) {
    return n->kind() == aten::append &&
        n->inputs().at(0)->node()->kind() == prim::ListConstruct;
  }

  bool tryMakeCreationAndMutationAtomic(
      Value* mutated_value,
      Node* mutating_op) {
    // We can only remove mutation to values that are unique aliases in the
    // graph. if x = y[0] or y = self.y, then removing the mutation could
    // change observable semantics
    if (!newMemoryLocation(mutated_value)) {
      return false;
    }

    // In order to safely remove a mutation, the creation of a tensor and its
    // subsequent mutation need to be one atomic operation
    return aliasDb_->moveBeforeTopologicallyValid(
        mutated_value->node(), mutating_op);
  }

  bool tryMakeUnaliasedIfOutputAndMutationAtomic(
      Value* mutated_value,
      Node* mutating_op) {
    // if cond:
    //    x = op()
    // else:
    //    x = op()
    // x = add_(1)
    // if x in both blocks have no other uses and are unaliased in the graph,
    // and we make the if node and the mutation atomic,
    // then removing mutation add_ does not change observable semantics

    if (mutated_value->node()->kind() != prim::If) {
      return false;
    }

    auto if_node = mutated_value->node();
    auto offset = mutated_value->offset();
    auto true_value = if_node->blocks().at(0)->outputs().at(offset);
    auto false_value = if_node->blocks().at(1)->outputs().at(offset);

    if (true_value->uses().size() > 1 || false_value->uses().size() > 1) {
      return false;
    }

    if (!newMemoryLocation(true_value) || !newMemoryLocation(false_value)) {
      return false;
    }

    return aliasDb_->moveBeforeTopologicallyValid(if_node, mutating_op);
  }

  void RemoveListMutation(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      auto* node = *it;
      it++;

      for (Block* sub_block : node->blocks()) {
        RemoveListMutation(sub_block);
      }

      if (!listAppendFollowingListConstruct(node)) {
        continue;
      }

      Value* mutated_value = node->inputs().at(0);
      if (!tryMakeCreationAndMutationAtomic(mutated_value, node)) {
        continue;
      }

      // We rewrite something like:
      // x = {v0}
      // x.append(v1)
      // to:
      // x = {v0, v1}
      // We can remove x.append from the the alias db list of writes.
      // All other aliasing properties remain valid.
      Node* list_construct = mutated_value->node();
      list_construct->addInput(node->inputs().at(1));
      node->output()->replaceAllUsesWith(mutated_value);
      aliasDb_->writeIndex_->erase(node);
      node->destroy();

      // TODO: don't strictly need to reset write cache, evaluate on models
      aliasDb_->writtenToLocationsIndex_ =
          aliasDb_->buildWrittenToLocationsIndex();
    }
  }

  void RemoveTensorMutation(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      auto* node = *it;
      it++;

      for (Block* sub_block : node->blocks()) {
        RemoveTensorMutation(sub_block);
      }

      // TODO: out op variants
      if (!inplaceOpVariant(node)) {
        continue;
      }

      Value* mutated_value = node->inputs().at(0);
      if (!tryMakeCreationAndMutationAtomic(mutated_value, node) &&
          !tryMakeUnaliasedIfOutputAndMutationAtomic(mutated_value, node)) {
        continue;
      }

      auto schema_name = node->schema().name();
      auto new_schema = schema_name.substr(0, schema_name.size() - 1);
      auto new_node = graph_->create(Symbol::fromQualString(new_schema), 1);
      new_node->copyMetadata(node);
      new_node->insertBefore(node);
      for (Value* input : node->inputs()) {
        new_node->addInput(input);
      }
      new_node->output()->setType(node->output()->type());
      mutated_value->replaceAllUsesAfterNodeWith(node, new_node->output());
      node->output()->replaceAllUsesWith(new_node->output());

      // We rewrite something like:
      // x = torch.zeros()
      // x.add_(1)
      // x.add_(2)
      // to:
      // x = torch.zeros()
      // x0 = x.add(1)
      // x0.add_(2)
      // For the remainder of the function, x0 will have the
      // same aliasing relationships as the original x.
      // To avoid rebuilding the entire alias db, we can replace
      // the memory dag element of x with x0.
      aliasDb_->replaceWithNewValue(mutated_value, new_node->output());

      // it is an invariant that all mutable types have an element in the memory
      // dag so we must regive x an alias db element. We have already verified
      // that the mutated value is a fresh alias with a single use.
      aliasDb_->createValue(mutated_value);

      // We must erase the destroyed node from the AliasDb lists of writes
      aliasDb_->writeIndex_->erase(node);
      node->destroy();

      // now that we have removed a mutating op, the write cache is stale
      // TODO: don't strictly need to reset write cache, evaluate on models
      aliasDb_->writtenToLocationsIndex_ =
          aliasDb_->buildWrittenToLocationsIndex();
    }
  }

 private:
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  std::shared_ptr<Graph> graph_;
};

void CreateFunctionalGraphs2(const std::shared_ptr<Graph>& graph) {
  // Run Constant Pooling so constants get hoisted
//   ConstantPooling(graph);
  FunctionalGraphSlicer2 func(graph);
  func.run();
  // Creation of Functional Subgraphs & Deinlining creates excess constants
//   ConstantPooling(graph);
}

void InlineFunctionalGraphs2(const std::shared_ptr<Graph>& graph) {
  InlineFunctionalGraphs(graph->block());
}

void RemoveListMutation2(const std::shared_ptr<Graph>& graph) {
  MutationRemover mr(graph);
  mr.removeListMutation();
}

void RemoveTensorMutation2(const std::shared_ptr<Graph>& graph) {
  MutationRemover mr(graph);
  mr.removeTensorMutation();
}



/////////////////////////////////////////////////////////////////////////

Operation createTensorExprOp(const Node* node) {
  auto kernel =
      std::make_shared<tensorexpr::TensorExprKernel>(node->g(attr::Subgraph));
  return [kernel](Stack& stack) {
//     std::cerr << "111 RUNNING COMPILED KERNEL!\n";
    RECORD_FUNCTION("TensorExpr", std::vector<c10::IValue>());
    if (!tensorexpr::fallbackAllowed()) {
//       std::cerr << "222 RUNNING COMPILED KERNEL!\n";
      kernel->run(stack);
      return 0;
    }

    try {
//       std::cerr << "RUNNING COMPILED KERNEL!\n";
      kernel->run(stack);
    } catch (const std::runtime_error& e) {
      kernel->fallback(stack);
    }
    return 0;
  };
}

RegisterOperators TensorExprOps({
    torch::jit::Operator(
        getTensorExprSymbol(),
        createTensorExprOp,
        AliasAnalysisKind::PURE_FUNCTION),
});

} // namespace jit
} // namespace torch
