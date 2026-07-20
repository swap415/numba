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
  %0 = icmp eq i64 %arg.arr.2, 1
  br i1 %0, label %B50.endif.us.epil.preheader, label %B50.endif.us.preheader.new

B50.endif.us.preheader.new:                       ; preds = %B50.endif.us.preheader
  %unroll_iter = and i64 %arg.arr.2, 9223372036854775806
  %xtraiter = and i64 %.arg.x, 7
  br label %B50.endif.us

B50.endif.us:                                     ; preds = %B74.B46.loopexit_crit_edge.us.1, %B50.endif.us.preheader.new
  %.115132138.us = phi i64 [ 0, %B50.endif.us.preheader.new ], [ %.122.us.1, %B74.B46.loopexit_crit_edge.us.1 ]
  %1 = icmp eq i64 %xtraiter, 0
  %2 = ptrtoint ptr %arg.arr.4 to i64
  %.295.us = mul i64 %.115132138.us, %arg.arr.6.0
  %.297.us = add i64 %.295.us, %2
  %.298.us = inttoptr i64 %.297.us to ptr
  %.298.promoted.us = load double, ptr %.298.us, align 8
  br i1 %1, label %B78.us.prol.loopexit, label %B78.us.prol.preheader

B78.us.prol.preheader:                            ; preds = %B50.endif.us
  br label %B78.us.prol

B78.us.prol:                                      ; preds = %B78.us.prol.preheader, %B78.us.prol
  %.300130.us.prol = phi double [ %.300.us.prol, %B78.us.prol ], [ %.298.promoted.us, %B78.us.prol.preheader ]
  %prol.iter = phi i64 [ %prol.iter.next, %B78.us.prol ], [ 0, %B78.us.prol.preheader ]
  %.300.us.prol = fadd double %.300130.us.prol, 1.000000e+00
  %prol.iter.next = add i64 %prol.iter, 1
  %prol.iter.cmp.not = icmp eq i64 %xtraiter, %prol.iter.next
  br i1 %prol.iter.cmp.not, label %B78.us.prol.loopexit.loopexit, label %B78.us.prol, !llvm.loop !0

B78.us.prol.loopexit.loopexit:                    ; preds = %B78.us.prol
  %3 = sub i64 %.arg.x, %prol.iter.next
  br label %B78.us.prol.loopexit

B78.us.prol.loopexit:                             ; preds = %B78.us.prol.loopexit.loopexit, %B50.endif.us
  %.300130.us.unr = phi double [ %.298.promoted.us, %B50.endif.us ], [ %.300.us.prol, %B78.us.prol.loopexit.loopexit ]
  %.229124128.us.unr = phi i64 [ %.arg.x, %B50.endif.us ], [ %3, %B78.us.prol.loopexit.loopexit ]
  %.300.us.lcssa.unr = phi double [ poison, %B50.endif.us ], [ %.300.us.prol, %B78.us.prol.loopexit.loopexit ]
  %4 = icmp slt i64 %arg.x, 8
  br i1 %4, label %B74.B46.loopexit_crit_edge.us, label %B78.us.preheader

B78.us.preheader:                                 ; preds = %B78.us.prol.loopexit
  %5 = add i64 %.229124128.us.unr, 8
  br label %B78.us

B78.us:                                           ; preds = %B78.us.preheader, %B78.us
  %lsr.iv5 = phi i64 [ %5, %B78.us.preheader ], [ %lsr.iv.next6, %B78.us ]
  %.300130.us = phi double [ %.300.us.7, %B78.us ], [ %.300130.us.unr, %B78.us.preheader ]
  %.300.us = fadd double %.300130.us, 1.000000e+00
  %.300.us.1 = fadd double %.300.us, 1.000000e+00
  %.300.us.2 = fadd double %.300.us.1, 1.000000e+00
  %.300.us.3 = fadd double %.300.us.2, 1.000000e+00
  %.300.us.4 = fadd double %.300.us.3, 1.000000e+00
  %.300.us.5 = fadd double %.300.us.4, 1.000000e+00
  %.300.us.6 = fadd double %.300.us.5, 1.000000e+00
  %.300.us.7 = fadd double %.300.us.6, 1.000000e+00
  %lsr.iv.next6 = add i64 %lsr.iv5, -8
  %.230.us.7 = icmp sgt i64 %lsr.iv.next6, 8
  br i1 %.230.us.7, label %B78.us, label %B74.B46.loopexit_crit_edge.us

B74.B46.loopexit_crit_edge.us:                    ; preds = %B78.us, %B78.us.prol.loopexit
  %.300.us.lcssa = phi double [ %.300.us.lcssa.unr, %B78.us.prol.loopexit ], [ %.300.us.7, %B78.us ]
  %6 = and i64 %.arg.x, 7
  %7 = icmp eq i64 %6, 0
  %8 = ptrtoint ptr %arg.arr.4 to i64
  %.122.us = or disjoint i64 %.115132138.us, 1
  %sunkaddr = getelementptr i8, ptr %arg.arr.4, i64 %.295.us
  store double %.300.us.lcssa, ptr %sunkaddr, align 8
  %.295.us.1 = mul i64 %.122.us, %arg.arr.6.0
  %.297.us.1 = add i64 %.295.us.1, %8
  %.298.us.1 = inttoptr i64 %.297.us.1 to ptr
  %.298.promoted.us.1 = load double, ptr %.298.us.1, align 8
  br i1 %7, label %B78.us.prol.loopexit.1, label %B78.us.prol.1.preheader

B78.us.prol.1.preheader:                          ; preds = %B74.B46.loopexit_crit_edge.us
  br label %B78.us.prol.1

B78.us.prol.1:                                    ; preds = %B78.us.prol.1.preheader, %B78.us.prol.1
  %.300130.us.prol.1 = phi double [ %.300.us.prol.1, %B78.us.prol.1 ], [ %.298.promoted.us.1, %B78.us.prol.1.preheader ]
  %prol.iter.1 = phi i64 [ %prol.iter.next.1, %B78.us.prol.1 ], [ 0, %B78.us.prol.1.preheader ]
  %.300.us.prol.1 = fadd double %.300130.us.prol.1, 1.000000e+00
  %prol.iter.next.1 = add i64 %prol.iter.1, 1
  %prol.iter.cmp.1.not = icmp eq i64 %xtraiter, %prol.iter.next.1
  br i1 %prol.iter.cmp.1.not, label %B78.us.prol.loopexit.1.loopexit, label %B78.us.prol.1, !llvm.loop !0

B78.us.prol.loopexit.1.loopexit:                  ; preds = %B78.us.prol.1
  %9 = sub i64 %.arg.x, %prol.iter.next.1
  br label %B78.us.prol.loopexit.1

B78.us.prol.loopexit.1:                           ; preds = %B78.us.prol.loopexit.1.loopexit, %B74.B46.loopexit_crit_edge.us
  %.300130.us.unr.1 = phi double [ %.298.promoted.us.1, %B74.B46.loopexit_crit_edge.us ], [ %.300.us.prol.1, %B78.us.prol.loopexit.1.loopexit ]
  %.229124128.us.unr.1 = phi i64 [ %.arg.x, %B74.B46.loopexit_crit_edge.us ], [ %9, %B78.us.prol.loopexit.1.loopexit ]
  %.300.us.lcssa.unr.1 = phi double [ poison, %B74.B46.loopexit_crit_edge.us ], [ %.300.us.prol.1, %B78.us.prol.loopexit.1.loopexit ]
  %10 = icmp slt i64 %arg.x, 8
  br i1 %10, label %B74.B46.loopexit_crit_edge.us.1, label %B78.us.1.preheader

B78.us.1.preheader:                               ; preds = %B78.us.prol.loopexit.1
  %11 = add i64 %.229124128.us.unr.1, 8
  br label %B78.us.1

B78.us.1:                                         ; preds = %B78.us.1.preheader, %B78.us.1
  %lsr.iv7 = phi i64 [ %11, %B78.us.1.preheader ], [ %lsr.iv.next8, %B78.us.1 ]
  %.300130.us.1 = phi double [ %.300.us.7.1, %B78.us.1 ], [ %.300130.us.unr.1, %B78.us.1.preheader ]
  %.300.us.14 = fadd double %.300130.us.1, 1.000000e+00
  %.300.us.1.1 = fadd double %.300.us.14, 1.000000e+00
  %.300.us.2.1 = fadd double %.300.us.1.1, 1.000000e+00
  %.300.us.3.1 = fadd double %.300.us.2.1, 1.000000e+00
  %.300.us.4.1 = fadd double %.300.us.3.1, 1.000000e+00
  %.300.us.5.1 = fadd double %.300.us.4.1, 1.000000e+00
  %.300.us.6.1 = fadd double %.300.us.5.1, 1.000000e+00
  %.300.us.7.1 = fadd double %.300.us.6.1, 1.000000e+00
  %lsr.iv.next8 = add i64 %lsr.iv7, -8
  %.230.us.7.1 = icmp sgt i64 %lsr.iv.next8, 8
  br i1 %.230.us.7.1, label %B78.us.1, label %B74.B46.loopexit_crit_edge.us.1

B74.B46.loopexit_crit_edge.us.1:                  ; preds = %B78.us.1, %B78.us.prol.loopexit.1
  %.300.us.lcssa.1 = phi double [ %.300.us.lcssa.unr.1, %B78.us.prol.loopexit.1 ], [ %.300.us.7.1, %B78.us.1 ]
  %.122.us.1 = add nuw i64 %.115132138.us, 2
  %sunkaddr9 = getelementptr i8, ptr %arg.arr.4, i64 %.295.us.1
  store double %.300.us.lcssa.1, ptr %sunkaddr9, align 8
  %niter.ncmp.1 = icmp eq i64 %.122.us.1, %unroll_iter
  br i1 %niter.ncmp.1, label %B112.loopexit.unr-lcssa, label %B50.endif.us

B112.loopexit.unr-lcssa:                          ; preds = %B74.B46.loopexit_crit_edge.us.1
  %12 = and i64 %arg.arr.2, 1
  %lcmp.mod2.not = icmp eq i64 %12, 0
  br i1 %lcmp.mod2.not, label %B112, label %B50.endif.us.epil.preheader

B50.endif.us.epil.preheader:                      ; preds = %B112.loopexit.unr-lcssa, %B50.endif.us.preheader
  %.115132138.us.epil.init = phi i64 [ 0, %B50.endif.us.preheader ], [ %.122.us.1, %B112.loopexit.unr-lcssa ]
  %13 = ptrtoint ptr %arg.arr.4 to i64
  %.295.us.epil = mul i64 %.115132138.us.epil.init, %arg.arr.6.0
  %.297.us.epil = add i64 %.295.us.epil, %13
  %.298.us.epil = inttoptr i64 %.297.us.epil to ptr
  %.298.promoted.us.epil = load double, ptr %.298.us.epil, align 8
  %xtraiter.epil = and i64 %.arg.x, 7
  %lcmp.mod.epil.not = icmp eq i64 %xtraiter.epil, 0
  br i1 %lcmp.mod.epil.not, label %B78.us.prol.loopexit.epil, label %B78.us.prol.epil.preheader

B78.us.prol.epil.preheader:                       ; preds = %B50.endif.us.epil.preheader
  br label %B78.us.prol.epil

B78.us.prol.epil:                                 ; preds = %B78.us.prol.epil.preheader, %B78.us.prol.epil
  %.300130.us.prol.epil = phi double [ %.300.us.prol.epil, %B78.us.prol.epil ], [ %.298.promoted.us.epil, %B78.us.prol.epil.preheader ]
  %prol.iter.epil = phi i64 [ %prol.iter.next.epil, %B78.us.prol.epil ], [ 0, %B78.us.prol.epil.preheader ]
  %.300.us.prol.epil = fadd double %.300130.us.prol.epil, 1.000000e+00
  %prol.iter.next.epil = add i64 %prol.iter.epil, 1
  %prol.iter.cmp.epil.not = icmp eq i64 %xtraiter.epil, %prol.iter.next.epil
  br i1 %prol.iter.cmp.epil.not, label %B78.us.prol.loopexit.epil.loopexit, label %B78.us.prol.epil, !llvm.loop !0

B78.us.prol.loopexit.epil.loopexit:               ; preds = %B78.us.prol.epil
  %14 = sub i64 %.arg.x, %prol.iter.next.epil
  br label %B78.us.prol.loopexit.epil

B78.us.prol.loopexit.epil:                        ; preds = %B78.us.prol.loopexit.epil.loopexit, %B50.endif.us.epil.preheader
  %.300130.us.unr.epil = phi double [ %.298.promoted.us.epil, %B50.endif.us.epil.preheader ], [ %.300.us.prol.epil, %B78.us.prol.loopexit.epil.loopexit ]
  %.229124128.us.unr.epil = phi i64 [ %.arg.x, %B50.endif.us.epil.preheader ], [ %14, %B78.us.prol.loopexit.epil.loopexit ]
  %.300.us.lcssa.unr.epil = phi double [ poison, %B50.endif.us.epil.preheader ], [ %.300.us.prol.epil, %B78.us.prol.loopexit.epil.loopexit ]
  %15 = icmp slt i64 %arg.x, 8
  br i1 %15, label %B74.B46.loopexit_crit_edge.us.epil, label %B78.us.epil.preheader

B78.us.epil.preheader:                            ; preds = %B78.us.prol.loopexit.epil
  %16 = add i64 %.229124128.us.unr.epil, 8
  br label %B78.us.epil

B78.us.epil:                                      ; preds = %B78.us.epil.preheader, %B78.us.epil
  %lsr.iv = phi i64 [ %16, %B78.us.epil.preheader ], [ %lsr.iv.next, %B78.us.epil ]
  %.300130.us.epil = phi double [ %.300.us.7.epil, %B78.us.epil ], [ %.300130.us.unr.epil, %B78.us.epil.preheader ]
  %.300.us.epil = fadd double %.300130.us.epil, 1.000000e+00
  %.300.us.1.epil = fadd double %.300.us.epil, 1.000000e+00
  %.300.us.2.epil = fadd double %.300.us.1.epil, 1.000000e+00
  %.300.us.3.epil = fadd double %.300.us.2.epil, 1.000000e+00
  %.300.us.4.epil = fadd double %.300.us.3.epil, 1.000000e+00
  %.300.us.5.epil = fadd double %.300.us.4.epil, 1.000000e+00
  %.300.us.6.epil = fadd double %.300.us.5.epil, 1.000000e+00
  %.300.us.7.epil = fadd double %.300.us.6.epil, 1.000000e+00
  %lsr.iv.next = add i64 %lsr.iv, -8
  %.230.us.7.epil = icmp sgt i64 %lsr.iv.next, 8
  br i1 %.230.us.7.epil, label %B78.us.epil, label %B74.B46.loopexit_crit_edge.us.epil

B74.B46.loopexit_crit_edge.us.epil:               ; preds = %B78.us.epil, %B78.us.prol.loopexit.epil
  %.300.us.lcssa.epil = phi double [ %.300.us.lcssa.unr.epil, %B78.us.prol.loopexit.epil ], [ %.300.us.7.epil, %B78.us.epil ]
  %sunkaddr10 = getelementptr i8, ptr %arg.arr.4, i64 %.295.us.epil
  store double %.300.us.lcssa.epil, ptr %sunkaddr10, align 8
  br label %B112

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
  br i1 %.8, label %common.ret, label %entry.endif, !prof !2

common.ret:                                       ; preds = %entry.endif.endif.endif.thread, %arg0.err, %entry, %entry.endif.endif.endif.endif.endif.endif, %entry.endif.if
  %common.ret.op = phi ptr [ null, %arg0.err ], [ null, %entry ], [ null, %entry.endif.if ], [ null, %entry.endif.endif.endif.thread ], [ @_Py_NoneStruct, %entry.endif.endif.endif.endif.endif.endif ]
  ret ptr %common.ret.op

entry.endif:                                      ; preds = %entry
  %.12 = load ptr, ptr @_ZN08NumbaEnv8__main__10make_funcs12_3clocals_3e7add_forB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1A7mutable7alignedEx, align 8
  %.17 = icmp eq ptr %.12, null
  br i1 %.17, label %entry.endif.if, label %entry.endif.endif, !prof !2

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
  br i1 %.33, label %entry.endif.endif.endif.thread, label %entry.endif.endif.endif.endif, !prof !2

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
  br i1 %.48.not, label %entry.endif.endif.endif.endif.endif, label %entry.endif.endif.endif.endif.if, !prof !2

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
  br i1 %.56.not, label %entry.endif.endif.endif.endif.endif.endif, label %arg0.err, !prof !3

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
!2 = !{!"branch_weights", i32 1, i32 99}
!3 = !{!"branch_weights", i32 99, i32 1}
