#include <torch/csrc/jit/pass_manager.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

std::vector<Pass>& getCustomPasses() {
  static std::vector<Pass> passes;
  return passes;
}

RegisterPass::RegisterPass(Pass p) {
  getCustomPasses().emplace_back(std::move(p));
}

void runPass(Pass p, const std::string& name, std::shared_ptr<Graph>& graph) {
  //   if (is_enabled(name.c_str(), JitLoggingLevels::GRAPH_DUMP)) {
  {
    std::cerr << jit_log_prefix(name, ::c10::str("Before ", name, "\n"));
    std::cerr << jit_log_prefix(name, graph->toString());
  }
  std::cerr << "*** " << name << " ***\n";
  p(graph);
  if (is_enabled(name.c_str(), JitLoggingLevels::GRAPH_DUMP)) {
    std::cerr << jit_log_prefix(name, ::c10::str("After ", name, "\n"));
    std::cerr << jit_log_prefix(name, graph->toString());
  }
}

} // namespace jit
} // namespace torch
