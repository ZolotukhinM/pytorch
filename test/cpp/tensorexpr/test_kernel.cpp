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
#include <chrono>

namespace torch {
namespace jit {

using namespace torch::indexing;
using namespace torch::jit;
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

int lstm(
    const at::Tensor& input_1,
    const at::Tensor& hx_1,
    const at::Tensor& cx_1,
    const at::Tensor& wih_1,
    const at::Tensor& whh_1,
    const at::Tensor& bih_1,
    const at::Tensor& bhh_1) {
  auto inputs_1 = at::unbind(input_1, 0);
  int _12 = 100; // at::len(inputs_1);

  auto cy = cx_1;
  auto hy = hx_1;

  for (int i = 0; i < _12; i++) {
    auto _18 = inputs_1[i]; // at::__getitem__(inputs_1, i);
    auto _19 = at::t(wih_1);
    auto _20 = at::mm(_18, _19);
    auto _21 = at::t(whh_1);
    auto _22 = at::mm(hy, _21);
    auto _23 = at::add(_20, _22, 1);
    auto _24 = at::add(_23, bih_1, 1);
    auto _gates_1 = at::add(_24, bhh_1, 1);

    // auto _44, auto _45, auto _46, auto _47 = prim::ConstantChunk[chunks=4,
    // dim=1](%gates_1)
    auto result = at::chunk(_gates_1, 4, 1);
    auto _44 = result[0];
    auto _45 = result[1];
    auto _46 = result[2];
    auto _47 = result[3];

    auto _ingate_3 = at::sigmoid(_44);
    auto _forgetgate_3 = at::sigmoid(_45);
    auto _cellgate_3 = at::tanh(_46);
    auto _outgate_3 = at::sigmoid(_47);
    auto _35 = at::mul(_forgetgate_3, cy);
    auto _36 = at::mul(_ingate_3, _cellgate_3);
    auto cy = at::add(_35, _36, 1);
    auto _38 = at::tanh(cy);
    auto hy = at::mul(_outgate_3, _38);
  }
  return hy.sizes()[0] + cy.sizes()[0];
}

void testKernel_3() {
  auto input_1 = at::zeros({100, 64, 512}, TensorOptions(kCUDA).dtype(at::kFloat));
  auto hx_1 = at::zeros({64, 512}, TensorOptions(kCUDA).dtype(at::kFloat));
  auto cx_1 = at::zeros({64, 512}, TensorOptions(kCUDA).dtype(at::kFloat));
  auto wih_1 = at::zeros({2048, 512}, TensorOptions(kCUDA).dtype(at::kFloat));
  auto whh_1 = at::zeros({2048, 512}, TensorOptions(kCUDA).dtype(at::kFloat));
  auto bih_1 = at::zeros({2048}, TensorOptions(kCUDA).dtype(at::kFloat));
  auto bhh_1 = at::zeros({2048}, TensorOptions(kCUDA).dtype(at::kFloat));
  int r = 0;
  // warmup
  for (int i = 0; i < 10; i++) {
    r += lstm(input_1, hx_1, cx_1, wih_1, whh_1, bih_1, bhh_1);
  }
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 50; i++) {
    r += lstm(input_1, hx_1, cx_1, wih_1, whh_1, bih_1, bhh_1);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto us = std::chrono::duration<double>(end - start).count() * 1e3;
  std::cerr << "CHECKSUM: " << r << "\n";
  std::cerr << "Time: " << us << "us\n";

#if 0
  bool _10 = true;
  int _9 = 0;
  int _8 = 1;
  auto inputs_1 = at::unbind(input_1, _9);
  int _12 = 100;//at::len(inputs_1);

  auto cy = cx_1;
  auto hy = hx_1;

  for (int i = 0; i < _12; i++) {
    auto _18 = inputs_1[i]; //at::__getitem__(inputs_1, i);
    auto _19 = at::t(wih_1);
    auto _20 = at::mm(_18, _19);
    auto _21 = at::t(whh_1);
    auto _22 = at::mm(hy, _21);
    auto _23 = at::add(_20, _22, _8);
    auto _24 = at::add(_23, bih_1, _8);
    auto _gates_1 = at::add(_24, bhh_1, _8);


    //auto _44, auto _45, auto _46, auto _47 = prim::ConstantChunk[chunks=4, dim=1](%gates_1)
    auto result = at::chunk(_gates_1, 4, 1);
    auto _44 = result[0];
    auto _45 = result[1];
    auto _46 = result[2];
    auto _47 = result[3];

    auto _ingate_3 = at::sigmoid(_44);
    auto _forgetgate_3 = at::sigmoid(_45);
    auto _cellgate_3 = at::tanh(_46);
    auto _outgate_3 = at::sigmoid(_47);
    auto _35 = at::mul(_forgetgate_3, cy);
    auto _36 = at::mul(_ingate_3, _cellgate_3);
    auto cy = at::add(_35, _36, _8);
    auto _38 = at::tanh(cy);
    auto hy = at::mul(_outgate_3, _38);
  }
  std::cerr << hy.shape()[0];
#endif
  /*

graph(%input_1 : Tensor, # Float(100:32768, 64:512, 512:1)
      %hx_1 : Tensor,    # Float(64:512, 512:1)
      %cx_1 : Tensor,    # Float(64:512, 512:1)
      %wih_1 : Tensor,   # Float(2048:512, 512:1)
      %whh_1 : Tensor,   # Float(2048:512, 512:1)
      %bih_1 : Tensor,   # Float(2048:1)
      %bhh_1 : Tensor):  # Float(2048:1)
  %10 : bool = prim::Constant[value=1]() # 2_py:28:4
  %9 : int = prim::Constant[value=0]() # 2_py:27:26
  %8 : int = prim::Constant[value=1]() # 2_py:11:59
  %inputs_1 : Tensor[] = aten::unbind(%input_1, %9) # 2_py:27:13
  %12 : int = aten::len(%inputs_1) # 2_py:28:25
  %hy : Tensor, %cy : Tensor = prim::Loop(%12, %10, %hx_1, %cx_1) # 2_py:28:4
    block0(%seq_idx_1 : int, %hy_5 : Tensor, %cy_5 : Tensor):
      %18 : Tensor = aten::__getitem__(%inputs_1, %seq_idx_1) # 2_py:29:32
      %19 : Tensor = aten::t(%wih_1) # 2_py:9:28
      %20 : Tensor = aten::mm(%18, %19) # 2_py:9:12
      %21 : Tensor = aten::t(%whh_1) # 2_py:9:53
      %22 : Tensor = aten::mm(%hy_5, %21) # 2_py:9:40
      %23 : Tensor = aten::add(%20, %22, %8) # 2_py:9:12
      %24 : Tensor = aten::add(%23, %bih_1, %8) # 2_py:9:12
      %gates_1 : Tensor = aten::add(%24, %bhh_1, %8) # 2_py:9:12
      %44 : Tensor, %45 : Tensor, %46 : Tensor, %47 : Tensor = prim::ConstantChunk[chunks=4, dim=1](%gates_1)
      %ingate_3 : Tensor = aten::sigmoid(%44) # 2_py:13:13
      %forgetgate_3 : Tensor = aten::sigmoid(%45) # 2_py:14:17
      %cellgate_3 : Tensor = aten::tanh(%46) # 2_py:15:15
      %outgate_3 : Tensor = aten::sigmoid(%47) # 2_py:16:14
      %35 : Tensor = aten::mul(%forgetgate_3, %cy_5) # 2_py:18:10
      %36 : Tensor = aten::mul(%ingate_3, %cellgate_3) # 2_py:18:30
      %cy_1 : Tensor = aten::add(%35, %36, %8) # 2_py:18:10
      %38 : Tensor = aten::tanh(%cy_1) # 2_py:19:19
      %hy_1 : Tensor = aten::mul(%outgate_3, %38) # 2_py:19:9
      -> (%10, %hy_1, %cy_1)
  %43 : (Tensor, Tensor) = prim::TupleConstruct(%hy, %cy)
  return (%43)


   */
}

} // namespace jit
} // namespace torch
