#include <torch/csrc/jit/tensorexpr/ir.h>

#include <torch/csrc/jit/tensorexpr/buffer.h>

namespace torch {
namespace jit {
namespace tensorexpr {

static Dtype ChooseDtype(const Dtype& buffer_dtype, const Dtype& index_dtype) {
  return Dtype(buffer_dtype, index_dtype.lanes());
}

static Dtype dtypeOfIndices(const std::vector<const Expr*>& indices) {
  if (!indices.size()) {
    throw malformed_input();
  }
  Dtype dt = indices.at(0)->dtype();
  for (size_t i = 1; i < indices.size(); ++i) {
    if (indices.at(i)->dtype() != dt) {
      throw malformed_input();
    }
  }
  return dt;
}

static bool indicesValid(const std::vector<const Expr*>& indices) {
  if (indices.size() == 0) {
    return false;
  }
  Dtype index_dtype = dtypeOfIndices(indices);
  if (indices.size() > 1 && index_dtype.lanes() > 1) {
    // Multilane is only allowed in a flattened (i.e. 1D) index
    return false;
  }
  if (index_dtype.scalar_type() != ScalarType::Int) {
    return false;
  }
  return true;
}

Load::Load(
    const Buffer& buffer,
    const std::vector<const Expr*>& indices,
    const Expr* mask)
    : Load(
          ChooseDtype(buffer.dtype(), dtypeOfIndices(indices)),
          buffer.data(),
          indices,
          mask) {}

Load::Load(
    Dtype dtype,
    const Buf* buf,
    const std::vector<const Expr*>& indices,
    const Expr* mask)
    : ExprNodeBase(dtype), buf_(buf), indices_(indices), mask_(mask) {
  if (buf->base_handle()->dtype() != kHandle) {
    throw malformed_input();
  }
  if (!indicesValid(indices)) {
    throw malformed_input();
  }
  Dtype index_dtype = dtypeOfIndices(indices);
  if (index_dtype.lanes() != mask->dtype().lanes()) {
    throw malformed_input();
  }
}

ExprHandle Load::make(
    const Buffer& buffer,
    const std::vector<ExprHandle>& indices,
    const ExprHandle& mask) {
  return ExprHandle(
      new Load(buffer, ExprHandleVectorToExprVector(indices), mask.node()));
}
ExprHandle Load::make(
    Dtype dtype,
    const BufHandle& buf,
    const std::vector<ExprHandle>& indices,
    const ExprHandle& mask) {
  return ExprHandle(new Load(
      dtype, buf.node(), ExprHandleVectorToExprVector(indices), mask.node()));
}

Store::Store(
    const Buffer& buffer,
    const std::vector<const Expr*>& indices,
    const Expr* value,
    const Expr* mask)
    : Store(buffer.data(), indices, value, mask) {
  if (buffer.dtype().scalar_type() != value->dtype().scalar_type()) {
    throw malformed_input();
  }
}

Store::Store(
    const Buf* buf,
    std::vector<const Expr*> indices,
    const Expr* value,
    const Expr* mask)
    : buf_(buf), indices_(std::move(indices)), value_(value), mask_(mask) {
  if (buf->dtype() != kHandle) {
    throw malformed_input();
  }
  /*
  TODO: Reenable the checks.
  The reason they are disabled is that kernel.cpp is using Buffers somewhat
  loosely: we don't set dimensions properly and just construct index expressions
  directly. We should harden that part and then we'd be able to turn on these
  checks.

  if (!indicesValid(indices)) {
    throw malformed_input();
  }
  if (!mask || !value) {
    throw malformed_input();
  }
  Dtype index_dtype = dtypeOfIndices(indices);
  if (index_dtype.lanes() != mask->dtype().lanes()) {
    throw malformed_input();
  }
  if (index_dtype.lanes() != value->dtype().lanes()) {
    throw malformed_input();
  }
  */
}

Store* Store::make(
    const Buffer& buffer,
    const std::vector<ExprHandle>& indices,
    const ExprHandle& value,
    const ExprHandle& mask) {
  return new Store(
      buffer, ExprHandleVectorToExprVector(indices), value.node(), mask.node());
}

Store* Store::make(
    const BufHandle& buf,
    const std::vector<ExprHandle>& indices,
    const ExprHandle& value,
    const ExprHandle& mask) {
  return new Store(
      buf.node(),
      ExprHandleVectorToExprVector(indices),
      value.node(),
      mask.node());
}

Store* Store::make(
    const BufHandle& buf,
    const std::vector<ExprHandle>& indices,
    const ExprHandle& value) {
  return new Store(
      buf.node(),
      ExprHandleVectorToExprVector(indices),
      value.node(),
      ExprHandle(1).node());
}

const Expr* flatten_index(
    const std::vector<const Expr*>& dims,
    const std::vector<const Expr*>& indices) {
  // Handle already flattened indices first
  if (indices.size() == 1) {
    return indices[0];
  }

  size_t ndim = dims.size();
  if (ndim != indices.size()) {
    throw malformed_input();
  }
  if (ndim == 0) {
    return new IntImm(0);
  }
  std::vector<const Expr*> strides(ndim);
  // stride[0] = 1,
  // stride[i] = stride[i-1]*dims[i-1], i > 0
  strides[0] = new IntImm(1);
  for (size_t i = 1; i < ndim; i++) {
    strides[i] = new Mul(strides[i - 1], dims[i - 1]);
  }

  const Expr* total_index = new IntImm(0);
  for (size_t i = 0; i < ndim; i++) {
    total_index = new Add(total_index, new Mul(indices[i], strides[i]));
  }
  return total_index;
}

Dtype Intrinsics::IntrinsicsDtype(IntrinsicsOp op_type, Dtype dt1) {
  // TODO: check the op_type and make a real decision
  return dt1;
}

Dtype Intrinsics::IntrinsicsDtype(IntrinsicsOp op_type, Dtype dt1, Dtype dt2) {
  // TODO: check the op_type and make a real decision
  return dt1;
}

Dtype Intrinsics::IntrinsicsDtype(
    IntrinsicsOp op_type,
    const std::vector<const Expr*>& params) {
  // TODO: check the op_type an dmake a real decision
  if (params.size() == 0) {
    throw malformed_input();
  }

  return params[0]->dtype();
}

int Intrinsics::OpArgCount(IntrinsicsOp op_type) {
  switch (op_type) {
    case kSin:
    case kCos:
    case kTan:
    case kAsin:
    case kAcos:
    case kAtan:
    case kSinh:
    case kCosh:
    case kTanh:
    case kExp:
    case kExpm1:
    case kFabs:
    case kLog:
    case kLog2:
    case kLog10:
    case kLog1p:
    case kErf:
    case kErfc:
    case kSqrt:
    case kRsqrt:
    case kCeil:
    case kFloor:
    case kRound:
    case kTrunc:
    case kFrac:
    case kLgamma:
      return 1;
    case kRand:
      return 0;
    case kAtan2:
    case kFmod:
    case kPow:
    case kRemainder:
      return 2;
    default:
      throw std::runtime_error("invalid op_type: " + c10::to_string(op_type));
  }
}

std::vector<const Expr*> ExprHandleVectorToExprVector(
    const std::vector<ExprHandle>& v) {
  std::vector<const Expr*> result(v.size());
  for (size_t i = 0; i < v.size(); i++) {
    result[i] = v[i].node();
  }
  return result;
}

std::vector<ExprHandle> ExprVectorToExprHandleVector(
    const std::vector<const Expr*>& v) {
  std::vector<ExprHandle> result(v.size());
  for (size_t i = 0; i < v.size(); i++) {
    result[i] = ExprHandle(v[i]);
  }
  return result;
}

std::vector<const Var*> VarHandleVectorToVarVector(
    const std::vector<VarHandle>& v) {
  std::vector<const Var*> result(v.size());
  for (size_t i = 0; i < v.size(); i++) {
    result[i] = v[i].node();
  }
  return result;
}

std::vector<VarHandle> VarVectorToVarHandleVector(
    const std::vector<const Var*>& v) {
  std::vector<VarHandle> result(v.size());
  for (size_t i = 0; i < v.size(); i++) {
    result[i] = VarHandle(v[i]);
  }
  return result;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
