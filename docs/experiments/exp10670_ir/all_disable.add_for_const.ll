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
  br label %B50.endif.endif

B46.loopexit:                                     ; preds = %B78
  %.119 = add nuw nsw i64 %.112129134, 1
  %sunkaddr = getelementptr i8, ptr %arg.arr.4, i64 %.291
  store double %.296, ptr %sunkaddr, align 8
  %exitcond.not = icmp eq i64 %.119, %arg.arr.2
  br i1 %exitcond.not, label %B112, label %B50.endif.endif, !llvm.loop !0

B78:                                              ; preds = %B50.endif.endif, %B78
  %lsr.iv = phi i64 [ 6, %B50.endif.endif ], [ %lsr.iv.next, %B78 ]
  %.296124125 = phi double [ %.294.promoted, %B50.endif.endif ], [ %.296, %B78 ]
  %.296 = fadd double %.296124125, 1.000000e+00
  %lsr.iv.next = add nsw i64 %lsr.iv, -1
  %.226 = icmp samesign ugt i64 %lsr.iv.next, 1
  br i1 %.226, label %B78, label %B46.loopexit, !llvm.loop !2

B112:                                             ; preds = %B46.loopexit, %B0.endif
  store ptr null, ptr %retptr, align 8
  ret i32 0

B50.endif.endif:                                  ; preds = %B50.endif.endif.lr.ph, %B46.loopexit
  %.112129134 = phi i64 [ 0, %B50.endif.endif.lr.ph ], [ %.119, %B46.loopexit ]
  %0 = ptrtoint ptr %arg.arr.4 to i64
  %.291 = mul i64 %.112129134, %arg.arr.6.0
  %.293 = add i64 %.291, %0
  %.294 = inttoptr i64 %.293 to ptr
  %.294.promoted = load double, ptr %.294, align 8
  br label %B78
}

define noundef ptr @_ZN7cpython8__main__10make_funcs12_3clocals_3e13add_for_constB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedE(ptr readnone captures(none) %py_closure, ptr %py_args, ptr readnone captures(none) %py_kws) local_unnamed_addr {
entry:
  %.5 = alloca ptr, align 8
  %.6 = call i32 (ptr, ptr, i64, i64, ...) @PyArg_UnpackTuple(ptr %py_args, ptr nonnull @".const.make_funcs.<locals>.add_for_const", i64 1, i64 1, ptr nonnull %.5)
  %.7 = icmp eq i32 %.6, 0
  %.21 = alloca { ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] }, align 8
  %.43 = alloca ptr, align 8
  br i1 %.7, label %common.ret, label %entry.endif, !prof !3

common.ret:                                       ; preds = %entry.endif.endif.if, %entry, %entry.endif.endif.endif.endif, %entry.endif.if
  %common.ret.op = phi ptr [ @_Py_NoneStruct, %entry.endif.endif.endif.endif ], [ null, %entry ], [ null, %entry.endif.if ], [ null, %entry.endif.endif.if ]
  ret ptr %common.ret.op

entry.endif:                                      ; preds = %entry
  %.11 = load ptr, ptr @_ZN08NumbaEnv8__main__10make_funcs12_3clocals_3e13add_for_constB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedE, align 8
  %.16 = icmp eq ptr %.11, null
  br i1 %.16, label %entry.endif.if, label %entry.endif.endif, !prof !3

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
  br i1 %.32, label %entry.endif.endif.if, label %entry.endif.endif.endif.endif, !prof !3

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
  br i1 %.4, label %common.ret1, label %.3.endif, !prof !3

common.ret1:                                      ; preds = %.3, %.3.endif
  ret void

.3.endif:                                         ; preds = %.3
  fence release
  %0 = tail call i8 @llvm.x86.atomic.sub.cc.i64(ptr nonnull %.1, i64 1, i32 4)
  %1 = trunc i8 %0 to i1
  br i1 %1, label %.3.endif.if, label %common.ret1, !prof !3

.3.endif.if:                                      ; preds = %.3.endif
  fence acquire
  tail call void @NRT_MemInfo_call_dtor(ptr nonnull %.1)
  ret void
}

; Function Attrs: nounwind
declare i8 @llvm.x86.atomic.sub.cc.i64(ptr, i64, i32 immarg) #3

declare void @NRT_MemInfo_call_dtor(ptr) local_unnamed_addr

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #2 = { noinline }
attributes #3 = { nounwind }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
!2 = distinct !{!2, !1}
!3 = !{!"branch_weights", i32 1, i32 99}
