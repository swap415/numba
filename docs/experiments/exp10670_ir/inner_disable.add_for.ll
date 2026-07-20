; ModuleID = 'make_funcs.<locals>.add_for'
source_filename = "<string>"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@".const.make_funcs.<locals>.add_for" = internal constant [28 x i8] c"make_funcs.<locals>.add_for\00"
@_ZN08NumbaEnv8__main__10make_funcs12_3clocals_3e7add_forB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedEx = common local_unnamed_addr global ptr null
@".const.missing Environment: _ZN08NumbaEnv8__main__10make_funcs12_3clocals_3e7add_forB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedEx" = internal constant [156 x i8] c"missing Environment: _ZN08NumbaEnv8__main__10make_funcs12_3clocals_3e7add_forB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedEx\00"
@PyExc_TypeError = external global i8
@".const.can't unbox array from PyObject into native value.  The object maybe of a different type" = internal constant [89 x i8] c"can't unbox array from PyObject into native value.  The object maybe of a different type\00"
@_Py_NoneStruct = external global i8
@PyExc_RuntimeError = external global i8

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none)
define noundef i32 @_ZN8__main__10make_funcs12_3clocals_3e7add_forB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedEx(ptr noalias writeonly captures(none) %retptr, ptr noalias readnone captures(none) %excinfo, ptr readnone captures(none) %arg.arr.0, ptr readnone captures(none) %arg.arr.1, i64 %arg.arr.2, i64 %arg.arr.3, ptr %arg.arr.4, i64 %arg.arr.5.0, i64 %arg.arr.6.0, i64 %arg.x) local_unnamed_addr #0 {
B0.endif:
  %.109137.not = icmp slt i64 %arg.arr.2, 1
  br i1 %.109137.not, label %B112, label %B50.endif.lr.ph

B50.endif.lr.ph:                                  ; preds = %B0.endif
  %.arg.x = tail call i64 @llvm.smax.i64(i64 %arg.x, i64 0)
  %.230126.not = icmp slt i64 %arg.x, 1
  br i1 %.230126.not, label %B112, label %B50.endif.us.preheader

B50.endif.us.preheader:                           ; preds = %B50.endif.lr.ph
  %xtraiter = and i64 %arg.arr.2, 3
  %0 = icmp ult i64 %arg.arr.2, 4
  br i1 %0, label %B50.endif.us.epil.preheader, label %B50.endif.us.preheader.new

B50.endif.us.preheader.new:                       ; preds = %B50.endif.us.preheader
  %unroll_iter = and i64 %arg.arr.2, 9223372036854775804
  %1 = add nuw i64 %.arg.x, 1
  br label %B50.endif.us

B50.endif.us:                                     ; preds = %B74.B46.loopexit_crit_edge.us.3, %B50.endif.us.preheader.new
  %.115132138.us = phi i64 [ 0, %B50.endif.us.preheader.new ], [ %.122.us.3, %B74.B46.loopexit_crit_edge.us.3 ]
  %2 = ptrtoint ptr %arg.arr.4 to i64
  %.295.us = mul i64 %.115132138.us, %arg.arr.6.0
  %.297.us = add i64 %.295.us, %2
  %.298.us = inttoptr i64 %.297.us to ptr
  %.298.promoted.us = load double, ptr %.298.us, align 8
  br label %B78.us

B78.us:                                           ; preds = %B50.endif.us, %B78.us
  %lsr.iv2 = phi i64 [ %1, %B50.endif.us ], [ %lsr.iv.next3, %B78.us ]
  %.300130.us = phi double [ %.298.promoted.us, %B50.endif.us ], [ %.300.us, %B78.us ]
  %.300.us = fadd double %.300130.us, 1.000000e+00
  %lsr.iv.next3 = add i64 %lsr.iv2, -1
  %.230.us = icmp sgt i64 %lsr.iv.next3, 1
  br i1 %.230.us, label %B78.us, label %B74.B46.loopexit_crit_edge.us, !llvm.loop !0

B74.B46.loopexit_crit_edge.us:                    ; preds = %B78.us
  %3 = ptrtoint ptr %arg.arr.4 to i64
  %.122.us = or disjoint i64 %.115132138.us, 1
  %sunkaddr = getelementptr i8, ptr %arg.arr.4, i64 %.295.us
  store double %.300.us, ptr %sunkaddr, align 8
  %.295.us.1 = mul i64 %.122.us, %arg.arr.6.0
  %.297.us.1 = add i64 %.295.us.1, %3
  %.298.us.1 = inttoptr i64 %.297.us.1 to ptr
  %.298.promoted.us.1 = load double, ptr %.298.us.1, align 8
  br label %B78.us.1

B78.us.1:                                         ; preds = %B78.us.1, %B74.B46.loopexit_crit_edge.us
  %lsr.iv4 = phi i64 [ %lsr.iv.next5, %B78.us.1 ], [ %1, %B74.B46.loopexit_crit_edge.us ]
  %.300130.us.1 = phi double [ %.298.promoted.us.1, %B74.B46.loopexit_crit_edge.us ], [ %.300.us.1, %B78.us.1 ]
  %.300.us.1 = fadd double %.300130.us.1, 1.000000e+00
  %lsr.iv.next5 = add i64 %lsr.iv4, -1
  %.230.us.1 = icmp sgt i64 %lsr.iv.next5, 1
  br i1 %.230.us.1, label %B78.us.1, label %B74.B46.loopexit_crit_edge.us.1, !llvm.loop !0

B74.B46.loopexit_crit_edge.us.1:                  ; preds = %B78.us.1
  %4 = ptrtoint ptr %arg.arr.4 to i64
  %.122.us.1 = or disjoint i64 %.115132138.us, 2
  %sunkaddr10 = getelementptr i8, ptr %arg.arr.4, i64 %.295.us.1
  store double %.300.us.1, ptr %sunkaddr10, align 8
  %.295.us.2 = mul i64 %.122.us.1, %arg.arr.6.0
  %.297.us.2 = add i64 %.295.us.2, %4
  %.298.us.2 = inttoptr i64 %.297.us.2 to ptr
  %.298.promoted.us.2 = load double, ptr %.298.us.2, align 8
  br label %B78.us.2

B78.us.2:                                         ; preds = %B78.us.2, %B74.B46.loopexit_crit_edge.us.1
  %lsr.iv6 = phi i64 [ %lsr.iv.next7, %B78.us.2 ], [ %1, %B74.B46.loopexit_crit_edge.us.1 ]
  %.300130.us.2 = phi double [ %.298.promoted.us.2, %B74.B46.loopexit_crit_edge.us.1 ], [ %.300.us.2, %B78.us.2 ]
  %.300.us.2 = fadd double %.300130.us.2, 1.000000e+00
  %lsr.iv.next7 = add i64 %lsr.iv6, -1
  %.230.us.2 = icmp sgt i64 %lsr.iv.next7, 1
  br i1 %.230.us.2, label %B78.us.2, label %B74.B46.loopexit_crit_edge.us.2, !llvm.loop !0

B74.B46.loopexit_crit_edge.us.2:                  ; preds = %B78.us.2
  %5 = ptrtoint ptr %arg.arr.4 to i64
  %.122.us.2 = or disjoint i64 %.115132138.us, 3
  %sunkaddr11 = getelementptr i8, ptr %arg.arr.4, i64 %.295.us.2
  store double %.300.us.2, ptr %sunkaddr11, align 8
  %.295.us.3 = mul i64 %.122.us.2, %arg.arr.6.0
  %.297.us.3 = add i64 %.295.us.3, %5
  %.298.us.3 = inttoptr i64 %.297.us.3 to ptr
  %.298.promoted.us.3 = load double, ptr %.298.us.3, align 8
  br label %B78.us.3

B78.us.3:                                         ; preds = %B78.us.3, %B74.B46.loopexit_crit_edge.us.2
  %lsr.iv8 = phi i64 [ %lsr.iv.next9, %B78.us.3 ], [ %1, %B74.B46.loopexit_crit_edge.us.2 ]
  %.300130.us.3 = phi double [ %.298.promoted.us.3, %B74.B46.loopexit_crit_edge.us.2 ], [ %.300.us.3, %B78.us.3 ]
  %.300.us.3 = fadd double %.300130.us.3, 1.000000e+00
  %lsr.iv.next9 = add i64 %lsr.iv8, -1
  %.230.us.3 = icmp sgt i64 %lsr.iv.next9, 1
  br i1 %.230.us.3, label %B78.us.3, label %B74.B46.loopexit_crit_edge.us.3, !llvm.loop !0

B74.B46.loopexit_crit_edge.us.3:                  ; preds = %B78.us.3
  %.122.us.3 = add nuw nsw i64 %.115132138.us, 4
  %sunkaddr12 = getelementptr i8, ptr %arg.arr.4, i64 %.295.us.3
  store double %.300.us.3, ptr %sunkaddr12, align 8
  %niter.ncmp.3 = icmp eq i64 %.122.us.3, %unroll_iter
  br i1 %niter.ncmp.3, label %B112.loopexit.unr-lcssa, label %B50.endif.us

B112.loopexit.unr-lcssa:                          ; preds = %B74.B46.loopexit_crit_edge.us.3
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %B112, label %B50.endif.us.epil.preheader

B50.endif.us.epil.preheader:                      ; preds = %B112.loopexit.unr-lcssa, %B50.endif.us.preheader
  %.115132138.us.epil.init = phi i64 [ 0, %B50.endif.us.preheader ], [ %.122.us.3, %B112.loopexit.unr-lcssa ]
  %6 = add nuw i64 %.arg.x, 1
  br label %B50.endif.us.epil

B50.endif.us.epil:                                ; preds = %B74.B46.loopexit_crit_edge.us.epil, %B50.endif.us.epil.preheader
  %.115132138.us.epil = phi i64 [ %.122.us.epil, %B74.B46.loopexit_crit_edge.us.epil ], [ %.115132138.us.epil.init, %B50.endif.us.epil.preheader ]
  %epil.iter = phi i64 [ %epil.iter.next, %B74.B46.loopexit_crit_edge.us.epil ], [ 0, %B50.endif.us.epil.preheader ]
  %7 = ptrtoint ptr %arg.arr.4 to i64
  %.295.us.epil = mul i64 %.115132138.us.epil, %arg.arr.6.0
  %.297.us.epil = add i64 %.295.us.epil, %7
  %.298.us.epil = inttoptr i64 %.297.us.epil to ptr
  %.298.promoted.us.epil = load double, ptr %.298.us.epil, align 8
  br label %B78.us.epil

B78.us.epil:                                      ; preds = %B78.us.epil, %B50.endif.us.epil
  %lsr.iv = phi i64 [ %lsr.iv.next, %B78.us.epil ], [ %6, %B50.endif.us.epil ]
  %.300130.us.epil = phi double [ %.298.promoted.us.epil, %B50.endif.us.epil ], [ %.300.us.epil, %B78.us.epil ]
  %.300.us.epil = fadd double %.300130.us.epil, 1.000000e+00
  %lsr.iv.next = add i64 %lsr.iv, -1
  %.230.us.epil = icmp sgt i64 %lsr.iv.next, 1
  br i1 %.230.us.epil, label %B78.us.epil, label %B74.B46.loopexit_crit_edge.us.epil, !llvm.loop !0

B74.B46.loopexit_crit_edge.us.epil:               ; preds = %B78.us.epil
  %.122.us.epil = add nuw nsw i64 %.115132138.us.epil, 1
  %sunkaddr13 = getelementptr i8, ptr %arg.arr.4, i64 %.295.us.epil
  store double %.300.us.epil, ptr %sunkaddr13, align 8
  %epil.iter.next = add i64 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter
  br i1 %epil.iter.cmp.not, label %B112, label %B50.endif.us.epil, !llvm.loop !2

B112:                                             ; preds = %B74.B46.loopexit_crit_edge.us.epil, %B112.loopexit.unr-lcssa, %B50.endif.lr.ph, %B0.endif
  store ptr null, ptr %retptr, align 8
  ret i32 0
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #1

define noundef ptr @_ZN7cpython8__main__10make_funcs12_3clocals_3e7add_forB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedEx(ptr readnone captures(none) %py_closure, ptr %py_args, ptr readnone captures(none) %py_kws) local_unnamed_addr {
entry:
  %.5 = alloca ptr, align 8
  %.6 = alloca ptr, align 8
  %.7 = call i32 (ptr, ptr, i64, i64, ...) @PyArg_UnpackTuple(ptr %py_args, ptr nonnull @".const.make_funcs.<locals>.add_for", i64 2, i64 2, ptr nonnull %.5, ptr nonnull %.6)
  %.8 = icmp eq i32 %.7, 0
  %.22 = alloca { ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] }, align 8
  %.60 = alloca ptr, align 8
  br i1 %.8, label %common.ret, label %entry.endif, !prof !3

common.ret:                                       ; preds = %entry.endif.endif.endif.thread, %arg0.err, %entry, %entry.endif.endif.endif.endif.endif.endif, %entry.endif.if
  %common.ret.op = phi ptr [ null, %arg0.err ], [ null, %entry ], [ null, %entry.endif.if ], [ null, %entry.endif.endif.endif.thread ], [ @_Py_NoneStruct, %entry.endif.endif.endif.endif.endif.endif ]
  ret ptr %common.ret.op

entry.endif:                                      ; preds = %entry
  %.12 = load ptr, ptr @_ZN08NumbaEnv8__main__10make_funcs12_3clocals_3e7add_forB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedEx, align 8
  %.17 = icmp eq ptr %.12, null
  br i1 %.17, label %entry.endif.if, label %entry.endif.endif, !prof !3

entry.endif.if:                                   ; preds = %entry.endif
  call void @PyErr_SetString(ptr nonnull @PyExc_RuntimeError, ptr nonnull @".const.missing Environment: _ZN08NumbaEnv8__main__10make_funcs12_3clocals_3e7add_forB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedEx")
  br label %common.ret

entry.endif.endif:                                ; preds = %entry.endif
  %.21 = load ptr, ptr %.5, align 8
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(56) %.22, i8 0, i64 56, i1 false)
  %.26 = call i32 @NRT_adapt_ndarray_from_python(ptr %.21, ptr nonnull %.22)
  %sunkaddr = getelementptr inbounds i8, ptr %.22, i64 24
  %.30 = load i64, ptr %sunkaddr, align 8
  %.31 = icmp ne i64 %.30, 8
  %.32 = icmp ne i32 %.26, 0
  %.33 = or i1 %.32, %.31
  br i1 %.33, label %entry.endif.endif.endif.thread, label %entry.endif.endif.endif.endif, !prof !3

entry.endif.endif.endif.thread:                   ; preds = %entry.endif.endif
  call void @PyErr_SetString(ptr nonnull @PyExc_TypeError, ptr nonnull @".const.can't unbox array from PyObject into native value.  The object maybe of a different type")
  br label %common.ret

entry.endif.endif.endif.endif:                    ; preds = %entry.endif.endif
  %.37.fca.0.load = load ptr, ptr %.22, align 8
  %sunkaddr1 = getelementptr inbounds i8, ptr %.22, i64 16
  %.37.fca.2.load = load i64, ptr %sunkaddr1, align 8
  %sunkaddr2 = getelementptr inbounds i8, ptr %.22, i64 32
  %.37.fca.4.load = load ptr, ptr %sunkaddr2, align 8
  %sunkaddr3 = getelementptr inbounds i8, ptr %.22, i64 48
  %.37.fca.6.0.load = load i64, ptr %sunkaddr3, align 8
  %.44 = load ptr, ptr %.6, align 8
  %.47 = call ptr @PyNumber_Long(ptr %.44)
  %.48.not = icmp eq ptr %.47, null
  br i1 %.48.not, label %entry.endif.endif.endif.endif.endif, label %entry.endif.endif.endif.endif.if, !prof !3

arg0.err:                                         ; preds = %entry.endif.endif.endif.endif.endif
  call void @NRT_decref(ptr %.37.fca.0.load)
  br label %common.ret

entry.endif.endif.endif.endif.if:                 ; preds = %entry.endif.endif.endif.endif
  %.50 = call i64 @PyLong_AsLongLong(ptr nonnull %.47)
  call void @Py_DecRef(ptr nonnull %.47)
  br label %entry.endif.endif.endif.endif.endif

entry.endif.endif.endif.endif.endif:              ; preds = %entry.endif.endif.endif.endif.if, %entry.endif.endif.endif.endif
  %.45.0 = phi i64 [ %.50, %entry.endif.endif.endif.endif.if ], [ 0, %entry.endif.endif.endif.endif ]
  %.55 = call ptr @PyErr_Occurred()
  %.56.not = icmp eq ptr %.55, null
  br i1 %.56.not, label %entry.endif.endif.endif.endif.endif.endif, label %arg0.err, !prof !4

entry.endif.endif.endif.endif.endif.endif:        ; preds = %entry.endif.endif.endif.endif.endif
  %.66 = call i32 @_ZN8__main__10make_funcs12_3clocals_3e7add_forB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedEx(ptr nonnull %.60, ptr nonnull poison, ptr poison, ptr poison, i64 %.37.fca.2.load, i64 poison, ptr %.37.fca.4.load, i64 poison, i64 %.37.fca.6.0.load, i64 %.45.0) #3
  call void @NRT_decref(ptr %.37.fca.0.load)
  call void @Py_IncRef(ptr nonnull @_Py_NoneStruct)
  br label %common.ret
}

declare i32 @PyArg_UnpackTuple(ptr, ptr, i64, i64, ...) local_unnamed_addr

declare void @PyErr_SetString(ptr, ptr) local_unnamed_addr

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #2

declare i32 @NRT_adapt_ndarray_from_python(ptr captures(none), ptr captures(none)) local_unnamed_addr

declare ptr @PyNumber_Long(ptr) local_unnamed_addr

declare i64 @PyLong_AsLongLong(ptr) local_unnamed_addr

declare void @Py_DecRef(ptr) local_unnamed_addr

declare ptr @PyErr_Occurred() local_unnamed_addr

declare void @Py_IncRef(ptr) local_unnamed_addr

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none)
define ptr @cfunc._ZN8__main__10make_funcs12_3clocals_3e7add_forB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedEx({ ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] } %.1, i64 %.2) local_unnamed_addr #0 {
entry:
  %.4 = alloca ptr, align 8
  store ptr null, ptr %.4, align 8
  %extracted.nitems = extractvalue { ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] } %.1, 2
  %extracted.data = extractvalue { ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] } %.1, 4
  %extracted.strides = extractvalue { ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] } %.1, 6
  %.9 = extractvalue [1 x i64] %extracted.strides, 0
  %.10 = call i32 @_ZN8__main__10make_funcs12_3clocals_3e7add_forB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedEx(ptr nonnull %.4, ptr nonnull poison, ptr poison, ptr poison, i64 %extracted.nitems, i64 poison, ptr %extracted.data, i64 poison, i64 %.9, i64 %.2) #3
  %.20 = load ptr, ptr %.4, align 8
  ret ptr %.20
}

; Function Attrs: noinline
define linkonce_odr void @NRT_decref(ptr %.1) local_unnamed_addr #3 {
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
declare i8 @llvm.x86.atomic.sub.cc.i64(ptr, i64, i32 immarg) #4

declare void @NRT_MemInfo_call_dtor(ptr) local_unnamed_addr

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #5

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #3 = { noinline }
attributes #4 = { nounwind }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
!2 = distinct !{!2, !1}
!3 = !{!"branch_weights", i32 1, i32 99}
!4 = !{!"branch_weights", i32 99, i32 1}
