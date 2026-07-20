; ModuleID = 'make_funcs.<locals>.add_for_const'
source_filename = "<string>"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@".const.make_funcs.<locals>.add_for_const" = internal constant [34 x i8] c"make_funcs.<locals>.add_for_const\00"
@_ZN08NumbaEnv8__main__10make_funcs12_3clocals_3e13add_for_constB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedE = common local_unnamed_addr global ptr null
@".const.missing Environment: _ZN08NumbaEnv8__main__10make_funcs12_3clocals_3e13add_for_constB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedE" = internal constant [162 x i8] c"missing Environment: _ZN08NumbaEnv8__main__10make_funcs12_3clocals_3e13add_for_constB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedE\00"
@PyExc_TypeError = external global i8
@".const.can't unbox array from PyObject into native value.  The object maybe of a different type" = internal constant [89 x i8] c"can't unbox array from PyObject into native value.  The object maybe of a different type\00"
@_Py_NoneStruct = external global i8
@PyExc_RuntimeError = external global i8

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none)
define noundef i32 @_ZN8__main__10make_funcs12_3clocals_3e13add_for_constB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedE(ptr noalias writeonly captures(none) %retptr, ptr noalias readnone captures(none) %excinfo, ptr readnone captures(none) %arg.arr.0, ptr readnone captures(none) %arg.arr.1, i64 %arg.arr.2, i64 %arg.arr.3, ptr %arg.arr.4, i64 %arg.arr.5.0, i64 %arg.arr.6.0) local_unnamed_addr #0 {
B0.endif:
  %.106133.not = icmp slt i64 %arg.arr.2, 1
  br i1 %.106133.not, label %B112, label %B50.endif.endif.lr.ph

B50.endif.endif.lr.ph:                            ; preds = %B0.endif
  %xtraiter = and i64 %arg.arr.2, 3
  %0 = icmp ult i64 %arg.arr.2, 4
  br i1 %0, label %B50.endif.endif.epil.preheader, label %B50.endif.endif.lr.ph.new

B50.endif.endif.lr.ph.new:                        ; preds = %B50.endif.endif.lr.ph
  %1 = ptrtoint ptr %arg.arr.4 to i64
  %unroll_iter = and i64 %arg.arr.2, 9223372036854775804
  %2 = mul i64 %arg.arr.6.0, 3
  %3 = shl i64 %arg.arr.6.0, 2
  br label %B50.endif.endif

B112.loopexit.unr-lcssa:                          ; preds = %B50.endif.endif
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %B112, label %B50.endif.endif.epil.preheader

B50.endif.endif.epil.preheader:                   ; preds = %B112.loopexit.unr-lcssa, %B50.endif.endif.lr.ph
  %.112129134.epil.init = phi i64 [ 0, %B50.endif.endif.lr.ph ], [ %.119.3, %B112.loopexit.unr-lcssa ]
  %4 = ptrtoint ptr %arg.arr.4 to i64
  %5 = mul i64 %.112129134.epil.init, %arg.arr.6.0
  %6 = add i64 %4, %5
  br label %B50.endif.endif.epil

B50.endif.endif.epil:                             ; preds = %B50.endif.endif.epil, %B50.endif.endif.epil.preheader
  %lsr.iv5 = phi i64 [ %lsr.iv.next6, %B50.endif.endif.epil ], [ %xtraiter, %B50.endif.endif.epil.preheader ]
  %lsr.iv = phi i64 [ %lsr.iv.next, %B50.endif.endif.epil ], [ %6, %B50.endif.endif.epil.preheader ]
  %.294.epil = inttoptr i64 %lsr.iv to ptr
  %.294.promoted.epil = load double, ptr %.294.epil, align 8
  %.296.epil = fadd double %.294.promoted.epil, 1.000000e+00
  %.296.1.epil = fadd double %.296.epil, 1.000000e+00
  %.296.2.epil = fadd double %.296.1.epil, 1.000000e+00
  %.296.3.epil = fadd double %.296.2.epil, 1.000000e+00
  %.296.4.epil = fadd double %.296.3.epil, 1.000000e+00
  store double %.296.4.epil, ptr %.294.epil, align 8
  %lsr.iv.next = add i64 %lsr.iv, %arg.arr.6.0
  %lsr.iv.next6 = add nsw i64 %lsr.iv5, -1
  %epil.iter.cmp.not = icmp eq i64 %lsr.iv.next6, 0
  br i1 %epil.iter.cmp.not, label %B112, label %B50.endif.endif.epil, !llvm.loop !0

B112:                                             ; preds = %B50.endif.endif.epil, %B112.loopexit.unr-lcssa, %B0.endif
  store ptr null, ptr %retptr, align 8
  ret i32 0

B50.endif.endif:                                  ; preds = %B50.endif.endif, %B50.endif.endif.lr.ph.new
  %lsr.iv7 = phi i64 [ %lsr.iv.next8, %B50.endif.endif ], [ %1, %B50.endif.endif.lr.ph.new ]
  %.112129134 = phi i64 [ 0, %B50.endif.endif.lr.ph.new ], [ %.119.3, %B50.endif.endif ]
  %.294 = inttoptr i64 %lsr.iv7 to ptr
  %.294.promoted = load double, ptr %.294, align 8
  %.296 = fadd double %.294.promoted, 1.000000e+00
  %.296.1 = fadd double %.296, 1.000000e+00
  %.296.2 = fadd double %.296.1, 1.000000e+00
  %.296.3 = fadd double %.296.2, 1.000000e+00
  %.296.4 = fadd double %.296.3, 1.000000e+00
  store double %.296.4, ptr %.294, align 8
  %7 = add i64 %arg.arr.6.0, %lsr.iv7
  %.294.1 = inttoptr i64 %7 to ptr
  %.294.promoted.1 = load double, ptr %.294.1, align 8
  %.296.12 = fadd double %.294.promoted.1, 1.000000e+00
  %.296.1.1 = fadd double %.296.12, 1.000000e+00
  %.296.2.1 = fadd double %.296.1.1, 1.000000e+00
  %.296.3.1 = fadd double %.296.2.1, 1.000000e+00
  %.296.4.1 = fadd double %.296.3.1, 1.000000e+00
  store double %.296.4.1, ptr %.294.1, align 8
  %sunkaddr = inttoptr i64 %lsr.iv7 to ptr
  %sunkaddr9 = mul i64 %arg.arr.6.0, 2
  %sunkaddr10 = getelementptr i8, ptr %sunkaddr, i64 %sunkaddr9
  %.294.promoted.2 = load double, ptr %sunkaddr10, align 8
  %.296.23 = fadd double %.294.promoted.2, 1.000000e+00
  %.296.1.2 = fadd double %.296.23, 1.000000e+00
  %.296.2.2 = fadd double %.296.1.2, 1.000000e+00
  %.296.3.2 = fadd double %.296.2.2, 1.000000e+00
  %.296.4.2 = fadd double %.296.3.2, 1.000000e+00
  store double %.296.4.2, ptr %sunkaddr10, align 8
  %8 = add i64 %2, %lsr.iv7
  %.294.3 = inttoptr i64 %8 to ptr
  %.294.promoted.3 = load double, ptr %.294.3, align 8
  %.296.34 = fadd double %.294.promoted.3, 1.000000e+00
  %.296.1.3 = fadd double %.296.34, 1.000000e+00
  %.296.2.3 = fadd double %.296.1.3, 1.000000e+00
  %.296.3.3 = fadd double %.296.2.3, 1.000000e+00
  %.296.4.3 = fadd double %.296.3.3, 1.000000e+00
  %.119.3 = add nuw nsw i64 %.112129134, 4
  store double %.296.4.3, ptr %.294.3, align 8
  %lsr.iv.next8 = add i64 %lsr.iv7, %3
  %niter.ncmp.3 = icmp eq i64 %unroll_iter, %.119.3
  br i1 %niter.ncmp.3, label %B112.loopexit.unr-lcssa, label %B50.endif.endif
}

define noundef ptr @_ZN7cpython8__main__10make_funcs12_3clocals_3e13add_for_constB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedE(ptr readnone captures(none) %py_closure, ptr %py_args, ptr readnone captures(none) %py_kws) local_unnamed_addr {
entry:
  %.5 = alloca ptr, align 8
  %.6 = call i32 (ptr, ptr, i64, i64, ...) @PyArg_UnpackTuple(ptr %py_args, ptr nonnull @".const.make_funcs.<locals>.add_for_const", i64 1, i64 1, ptr nonnull %.5)
  %.7 = icmp eq i32 %.6, 0
  %.21 = alloca { ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] }, align 8
  %.43 = alloca ptr, align 8
  br i1 %.7, label %common.ret, label %entry.endif, !prof !2

common.ret:                                       ; preds = %entry.endif.endif.if, %entry, %entry.endif.endif.endif.endif, %entry.endif.if
  %common.ret.op = phi ptr [ @_Py_NoneStruct, %entry.endif.endif.endif.endif ], [ null, %entry ], [ null, %entry.endif.if ], [ null, %entry.endif.endif.if ]
  ret ptr %common.ret.op

entry.endif:                                      ; preds = %entry
  %.11 = load ptr, ptr @_ZN08NumbaEnv8__main__10make_funcs12_3clocals_3e13add_for_constB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedE, align 8
  %.16 = icmp eq ptr %.11, null
  br i1 %.16, label %entry.endif.if, label %entry.endif.endif, !prof !2

entry.endif.if:                                   ; preds = %entry.endif
  call void @PyErr_SetString(ptr nonnull @PyExc_RuntimeError, ptr nonnull @".const.missing Environment: _ZN08NumbaEnv8__main__10make_funcs12_3clocals_3e13add_for_constB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedE")
  br label %common.ret

entry.endif.endif:                                ; preds = %entry.endif
  %.20 = load ptr, ptr %.5, align 8
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(56) %.21, i8 0, i64 56, i1 false)
  %.25 = call i32 @NRT_adapt_ndarray_from_python(ptr %.20, ptr nonnull %.21)
  %sunkaddr = getelementptr inbounds i8, ptr %.21, i64 24
  %.29 = load i64, ptr %sunkaddr, align 8
  %.30 = icmp ne i64 %.29, 8
  %.31 = icmp ne i32 %.25, 0
  %.32 = or i1 %.31, %.30
  br i1 %.32, label %entry.endif.endif.if, label %entry.endif.endif.endif.endif, !prof !2

entry.endif.endif.if:                             ; preds = %entry.endif.endif
  call void @PyErr_SetString(ptr nonnull @PyExc_TypeError, ptr nonnull @".const.can't unbox array from PyObject into native value.  The object maybe of a different type")
  br label %common.ret

entry.endif.endif.endif.endif:                    ; preds = %entry.endif.endif
  %.36.fca.0.load = load ptr, ptr %.21, align 8
  %sunkaddr1 = getelementptr inbounds i8, ptr %.21, i64 16
  %.36.fca.2.load = load i64, ptr %sunkaddr1, align 8
  %sunkaddr2 = getelementptr inbounds i8, ptr %.21, i64 32
  %.36.fca.4.load = load ptr, ptr %sunkaddr2, align 8
  %sunkaddr3 = getelementptr inbounds i8, ptr %.21, i64 48
  %.36.fca.6.0.load = load i64, ptr %sunkaddr3, align 8
  %.49 = call i32 @_ZN8__main__10make_funcs12_3clocals_3e13add_for_constB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedE(ptr nonnull %.43, ptr nonnull poison, ptr poison, ptr poison, i64 %.36.fca.2.load, i64 poison, ptr %.36.fca.4.load, i64 poison, i64 %.36.fca.6.0.load) #2
  call void @NRT_decref(ptr %.36.fca.0.load)
  call void @Py_IncRef(ptr nonnull @_Py_NoneStruct)
  br label %common.ret
}

declare i32 @PyArg_UnpackTuple(ptr, ptr, i64, i64, ...) local_unnamed_addr

declare void @PyErr_SetString(ptr, ptr) local_unnamed_addr

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #1

declare i32 @NRT_adapt_ndarray_from_python(ptr captures(none), ptr captures(none)) local_unnamed_addr

declare void @Py_IncRef(ptr) local_unnamed_addr

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none)
define ptr @cfunc._ZN8__main__10make_funcs12_3clocals_3e13add_for_constB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedE({ ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] } %.1) local_unnamed_addr #0 {
entry:
  %.3 = alloca ptr, align 8
  store ptr null, ptr %.3, align 8
  %extracted.nitems = extractvalue { ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] } %.1, 2
  %extracted.data = extractvalue { ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] } %.1, 4
  %extracted.strides = extractvalue { ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] } %.1, 6
  %.8 = extractvalue [1 x i64] %extracted.strides, 0
  %.9 = call i32 @_ZN8__main__10make_funcs12_3clocals_3e13add_for_constB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedE(ptr nonnull %.3, ptr nonnull poison, ptr poison, ptr poison, i64 %extracted.nitems, i64 poison, ptr %extracted.data, i64 poison, i64 %.8) #2
  %.19 = load ptr, ptr %.3, align 8
  ret ptr %.19
}

; Function Attrs: noinline
define linkonce_odr void @NRT_decref(ptr %.1) local_unnamed_addr #2 {
.3:
  %.4 = icmp eq ptr %.1, null
  br i1 %.4, label %common.ret1, label %.3.endif, !prof !2

common.ret1:                                      ; preds = %.3, %.3.endif
  ret void

.3.endif:                                         ; preds = %.3
  fence release
  %0 = tail call i8 @llvm.x86.atomic.sub.cc.i64(ptr nonnull %.1, i64 1, i32 4)
  %1 = trunc i8 %0 to i1
  br i1 %1, label %.3.endif.if, label %common.ret1, !prof !2

.3.endif.if:                                      ; preds = %.3.endif
  fence acquire
  tail call void @NRT_MemInfo_call_dtor(ptr nonnull %.1)
  ret void
}

; Function Attrs: nounwind
declare i8 @llvm.x86.atomic.sub.cc.i64(ptr, i64, i32 immarg) #3

declare void @NRT_MemInfo_call_dtor(ptr) local_unnamed_addr

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #4

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #2 = { noinline }
attributes #3 = { nounwind }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
!2 = !{!"branch_weights", i32 1, i32 99}
