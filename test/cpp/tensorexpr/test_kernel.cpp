#include <test/cpp/tensorexpr/test_base.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/tensorexpr/buffer.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/torch.h>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace torch {
namespace jit {

using namespace torch::indexing;
using namespace torch::jit::tensorexpr;

void testKernel_1() {
  KernelScope kernel_scope;

  const auto graph_string = R"IR(
      graph(%0 : Float(5:3,3:1),
            %1 : Float(5:3,3:1)):
        %2 : Float(5:3,3:1) = aten::mul(%0, %1)
        %3 : Float(5:3,3:1) = aten::mul(%0, %2)
        return (%3))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = a * (a * b);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {a, b};
  Stmt* s = k.getCodeGenStmt();
  // TODO: verify stmt

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  o = stack[0].toTensor();
  for (size_t i = 0; i < 5 * 3; i++) {
    CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}

void testKernel_2() {
  KernelScope kernel_scope;

  const auto graph_string = R"IR(
      graph(%0 : Float(5:3,3:1),
            %1 : Float(5:1,3:5)):
        %2 : Float(5:3,3:1) = aten::mul(%0, %1)
        %3 : Float(5:3,3:1) = aten::mul(%0, %2)
        return (%3))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b =
      at::rand({3, 5}, TensorOptions(kCPU).dtype(at::kFloat)).transpose(0, 1);
  auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = a * (a * b);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {a, b};
  Stmt* s = k.getCodeGenStmt();
  // TODO: verify stmt

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  o = stack[0].toTensor();
  for (size_t i = 0; i < 5 * 3; i++) {
    CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}

void testKernel_3() {
  {
    KernelScope kernel_scope;

    const auto graph_string = R"IR(
graph(%x : Float(36:17, 17:1),
      %y : Float(63:36, 36:1, 1:1),
      %z : Float(33:63, 63:1, 1:1, 1:1)):
  %3 : int = prim::Constant[value=1]()
  %v1 : Float(63:612, 36:17, 17:1) = aten::add(%x, %y, %3)
  %6 : Float(33:38556, 63:612, 36:17, 17:1) = aten::add(%v1, %z, %3)
  return (%6))IR";
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    auto x = at::rand({36, 17}, TensorOptions(kCPU).dtype(at::kFloat));
    auto y = at::rand({63, 36, 1}, TensorOptions(kCPU).dtype(at::kFloat));
    auto z = at::rand({33, 63, 1, 1}, TensorOptions(kCPU).dtype(at::kFloat));
    auto o = at::zeros({33, 63, 36, 17}, TensorOptions(kCPU).dtype(at::kFloat));
    auto ref = x + y + z;
    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {x, y, z};
    Stmt* s = k.getCodeGenStmt();
    // TODO: verify stmt

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    o = stack[0].toTensor();
    for (size_t i = 0; i < 100; i++) {
      CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    }
  }
  {
    KernelScope kernel_scope;

    const auto graph_string = R"IR(
      graph(%0 : Float(5:3,3:1),
            %1 : Float(5:12,3:2)):
        %2 : Float(5:3,3:1) = aten::mul(%0, %1)
        %3 : Float(5:3,3:1) = aten::mul(%0, %2)
        return (%3))IR";
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
    auto b = at::rand({10, 6}, TensorOptions(kCPU).dtype(at::kFloat))
                 .index({Slice(None, None, 2), Slice(None, None, 2)});
    auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
    auto ref = a * (a * b);
    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {a, b};
    Stmt* s = k.getCodeGenStmt();
    // TODO: verify stmt

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    o = stack[0].toTensor();
    for (size_t i = 0; i < 5 * 3; i++) {
      CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    }
  }
}

} // namespace jit
} // namespace torch
