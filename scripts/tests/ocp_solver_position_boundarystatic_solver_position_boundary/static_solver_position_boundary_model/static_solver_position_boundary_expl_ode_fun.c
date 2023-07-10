/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) static_solver_position_boundary_expl_ode_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[23] = {19, 1, 0, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
static const casadi_int casadi_s1[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s2[3] = {0, 0, 0};

/* static_solver_position_boundary_expl_ode_fun:(i0[19],i1,i2[])->(o0[19]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a4, a5, a6, a7, a8, a9;
  a0=2.;
  a1=arg[0]? arg[0][3] : 0;
  a2=casadi_sq(a1);
  a3=arg[0]? arg[0][4] : 0;
  a4=casadi_sq(a3);
  a2=(a2+a4);
  a2=(a0*a2);
  a4=1.;
  a2=(a2-a4);
  a5=6.3661977236758142e-06;
  a6=(a5*a2);
  a7=arg[0]? arg[0][7] : 0;
  a6=(a6*a7);
  a8=arg[0]? arg[0][5] : 0;
  a9=(a3*a8);
  a10=arg[0]? arg[0][6] : 0;
  a11=(a1*a10);
  a9=(a9+a11);
  a9=(a0*a9);
  a11=(a5*a9);
  a12=arg[0]? arg[0][8] : 0;
  a11=(a11*a12);
  a6=(a6+a11);
  a11=(a3*a10);
  a13=(a1*a8);
  a11=(a11-a13);
  a11=(a0*a11);
  a13=(a5*a11);
  a14=arg[0]? arg[0][9] : 0;
  a13=(a13*a14);
  a6=(a6+a13);
  a13=(a2*a6);
  a15=(a3*a8);
  a16=(a1*a10);
  a15=(a15-a16);
  a15=(a0*a15);
  a16=(a5*a15);
  a16=(a16*a7);
  a17=casadi_sq(a1);
  a18=casadi_sq(a8);
  a17=(a17+a18);
  a17=(a0*a17);
  a17=(a17-a4);
  a18=(a5*a17);
  a18=(a18*a12);
  a16=(a16+a18);
  a18=(a8*a10);
  a19=(a1*a3);
  a18=(a18+a19);
  a18=(a0*a18);
  a5=(a5*a18);
  a5=(a5*a14);
  a16=(a16+a5);
  a5=(a15*a16);
  a13=(a13+a5);
  a5=(a3*a10);
  a19=(a1*a8);
  a5=(a5+a19);
  a5=(a0*a5);
  a19=2.5464790894703257e-05;
  a20=(a19*a5);
  a20=(a20*a7);
  a21=(a8*a10);
  a22=(a1*a3);
  a21=(a21-a22);
  a21=(a0*a21);
  a22=(a19*a21);
  a22=(a22*a12);
  a20=(a20+a22);
  a22=casadi_sq(a1);
  a23=casadi_sq(a10);
  a22=(a22+a23);
  a0=(a0*a22);
  a0=(a0-a4);
  a19=(a19*a0);
  a19=(a19*a14);
  a20=(a20+a19);
  a20=(a20+a4);
  a19=(a5*a20);
  a13=(a13+a19);
  if (res[0]!=0) res[0][0]=a13;
  a19=(a9*a6);
  a22=(a17*a16);
  a19=(a19+a22);
  a22=(a21*a20);
  a19=(a19+a22);
  if (res[0]!=0) res[0][1]=a19;
  a22=(a11*a6);
  a23=(a18*a16);
  a22=(a22+a23);
  a23=(a0*a20);
  a22=(a22+a23);
  if (res[0]!=0) res[0][2]=a22;
  a23=1.0000000000000000e-02;
  a24=casadi_sq(a1);
  a25=casadi_sq(a3);
  a24=(a24+a25);
  a25=casadi_sq(a8);
  a24=(a24+a25);
  a25=casadi_sq(a10);
  a24=(a24+a25);
  a4=(a4-a24);
  a23=(a23*a4);
  a4=(a23*a1);
  a24=5.0000000000000000e-01;
  a25=4.0743665431525199e+02;
  a26=(a25*a2);
  a27=arg[0]? arg[0][10] : 0;
  a26=(a26*a27);
  a28=(a25*a9);
  a29=arg[0]? arg[0][11] : 0;
  a28=(a28*a29);
  a26=(a26+a28);
  a28=(a25*a11);
  a30=arg[0]? arg[0][12] : 0;
  a28=(a28*a30);
  a26=(a26+a28);
  a28=(a26*a3);
  a31=(a25*a15);
  a31=(a31*a27);
  a32=(a25*a17);
  a32=(a32*a29);
  a31=(a31+a32);
  a25=(a25*a18);
  a25=(a25*a30);
  a31=(a31+a25);
  a25=(a31*a8);
  a28=(a28+a25);
  a25=5.0929581789406498e+01;
  a32=(a25*a5);
  a32=(a32*a27);
  a27=(a25*a21);
  a27=(a27*a29);
  a32=(a32+a27);
  a25=(a25*a0);
  a25=(a25*a30);
  a32=(a32+a25);
  a25=(a32*a10);
  a28=(a28+a25);
  a28=(a24*a28);
  a4=(a4-a28);
  if (res[0]!=0) res[0][3]=a4;
  a4=(a26*a1);
  a28=(a32*a8);
  a4=(a4+a28);
  a28=(a31*a10);
  a4=(a4-a28);
  a4=(a24*a4);
  a28=(a23*a3);
  a4=(a4+a28);
  if (res[0]!=0) res[0][4]=a4;
  a4=(a31*a1);
  a28=(a32*a3);
  a4=(a4-a28);
  a28=(a26*a10);
  a4=(a4+a28);
  a4=(a24*a4);
  a28=(a23*a8);
  a4=(a4+a28);
  if (res[0]!=0) res[0][5]=a4;
  a1=(a32*a1);
  a3=(a31*a3);
  a1=(a1+a3);
  a8=(a26*a8);
  a1=(a1-a8);
  a24=(a24*a1);
  a23=(a23*a10);
  a24=(a24+a23);
  if (res[0]!=0) res[0][6]=a24;
  a24=arg[0]? arg[0][13] : 0;
  a23=(a9*a6);
  a10=(a17*a16);
  a23=(a23+a10);
  a10=(a21*a20);
  a23=(a23+a10);
  a10=(a2*a6);
  a1=(a15*a16);
  a10=(a10+a1);
  a1=(a5*a20);
  a10=(a10+a1);
  a1=(a23*a10);
  a1=(a24*a1);
  a8=(a17*a32);
  a3=(a21*a31);
  a8=(a8-a3);
  a8=(a8*a6);
  a3=(a21*a26);
  a4=(a9*a32);
  a3=(a3-a4);
  a3=(a3*a16);
  a8=(a8+a3);
  a3=(a9*a31);
  a4=(a17*a26);
  a3=(a3-a4);
  a3=(a3*a20);
  a8=(a8+a3);
  a1=(a1*a8);
  a3=(a11*a6);
  a4=(a18*a16);
  a3=(a3+a4);
  a4=(a0*a20);
  a3=(a3+a4);
  a4=casadi_sq(a3);
  a28=casadi_sq(a23);
  a4=(a4+a28);
  a4=(a24*a4);
  a28=(a15*a32);
  a25=(a5*a31);
  a28=(a28-a25);
  a28=(a28*a6);
  a25=(a5*a26);
  a30=(a2*a32);
  a25=(a25-a30);
  a25=(a25*a16);
  a28=(a28+a25);
  a25=(a2*a31);
  a30=(a15*a26);
  a25=(a25-a30);
  a25=(a25*a20);
  a28=(a28+a25);
  a4=(a4*a28);
  a1=(a1-a4);
  a4=(a3*a10);
  a4=(a24*a4);
  a25=(a18*a32);
  a30=(a0*a31);
  a25=(a25-a30);
  a25=(a25*a6);
  a6=(a0*a26);
  a30=(a11*a32);
  a6=(a6-a30);
  a6=(a6*a16);
  a25=(a25+a6);
  a6=(a11*a31);
  a16=(a18*a26);
  a6=(a6-a16);
  a6=(a6*a20);
  a25=(a25+a6);
  a4=(a4*a25);
  a1=(a1+a4);
  a4=casadi_sq(a10);
  a6=casadi_sq(a23);
  a4=(a4+a6);
  a6=casadi_sq(a3);
  a4=(a4+a6);
  a6=sqrt(a4);
  a6=(a6*a4);
  a1=(a1/a6);
  a4=arg[0]? arg[0][14] : 0;
  a20=(a23*a10);
  a20=(a4*a20);
  a20=(a20*a8);
  a16=casadi_sq(a3);
  a30=casadi_sq(a23);
  a16=(a16+a30);
  a16=(a4*a16);
  a16=(a16*a28);
  a20=(a20-a16);
  a16=(a3*a10);
  a16=(a4*a16);
  a16=(a16*a25);
  a20=(a20+a16);
  a16=casadi_sq(a10);
  a30=casadi_sq(a23);
  a16=(a16+a30);
  a30=casadi_sq(a3);
  a16=(a16+a30);
  a30=sqrt(a16);
  a30=(a30*a16);
  a20=(a20/a30);
  a1=(a1+a20);
  a20=arg[0]? arg[0][15] : 0;
  a16=(a23*a10);
  a16=(a20*a16);
  a16=(a16*a8);
  a27=casadi_sq(a3);
  a29=casadi_sq(a23);
  a27=(a27+a29);
  a27=(a20*a27);
  a27=(a27*a28);
  a16=(a16-a27);
  a27=(a3*a10);
  a27=(a20*a27);
  a27=(a27*a25);
  a16=(a16+a27);
  a27=casadi_sq(a10);
  a29=casadi_sq(a23);
  a27=(a27+a29);
  a29=casadi_sq(a3);
  a27=(a27+a29);
  a29=sqrt(a27);
  a29=(a29*a27);
  a16=(a16/a29);
  a1=(a1+a16);
  if (res[0]!=0) res[0][7]=a1;
  a1=(a10*a23);
  a1=(a24*a1);
  a1=(a1*a28);
  a16=casadi_sq(a3);
  a27=casadi_sq(a10);
  a16=(a16+a27);
  a16=(a24*a16);
  a16=(a16*a8);
  a1=(a1-a16);
  a16=(a3*a23);
  a16=(a24*a16);
  a16=(a16*a25);
  a1=(a1+a16);
  a1=(a1/a6);
  a16=(a10*a23);
  a16=(a4*a16);
  a16=(a16*a28);
  a27=casadi_sq(a3);
  a33=casadi_sq(a10);
  a27=(a27+a33);
  a27=(a4*a27);
  a27=(a27*a8);
  a16=(a16-a27);
  a27=(a3*a23);
  a27=(a4*a27);
  a27=(a27*a25);
  a16=(a16+a27);
  a16=(a16/a30);
  a1=(a1+a16);
  a16=(a10*a23);
  a16=(a20*a16);
  a16=(a16*a28);
  a27=casadi_sq(a3);
  a33=casadi_sq(a10);
  a27=(a27+a33);
  a27=(a20*a27);
  a27=(a27*a8);
  a16=(a16-a27);
  a27=(a3*a23);
  a27=(a20*a27);
  a27=(a27*a25);
  a16=(a16+a27);
  a16=(a16/a29);
  a1=(a1+a16);
  if (res[0]!=0) res[0][8]=a1;
  a1=6.1638047863431737e-02;
  a16=(a10*a3);
  a16=(a24*a16);
  a16=(a16*a28);
  a27=(a23*a3);
  a27=(a24*a27);
  a27=(a27*a8);
  a16=(a16+a27);
  a27=casadi_sq(a23);
  a33=casadi_sq(a10);
  a27=(a27+a33);
  a24=(a24*a27);
  a24=(a24*a25);
  a16=(a16-a24);
  a16=(a16/a6);
  a6=(a10*a3);
  a6=(a4*a6);
  a6=(a6*a28);
  a24=(a23*a3);
  a24=(a4*a24);
  a24=(a24*a8);
  a6=(a6+a24);
  a24=casadi_sq(a23);
  a27=casadi_sq(a10);
  a24=(a24+a27);
  a4=(a4*a24);
  a4=(a4*a25);
  a6=(a6-a4);
  a6=(a6/a30);
  a16=(a16+a6);
  a6=(a10*a3);
  a6=(a20*a6);
  a6=(a6*a28);
  a3=(a23*a3);
  a3=(a20*a3);
  a3=(a3*a8);
  a6=(a6+a3);
  a23=casadi_sq(a23);
  a10=casadi_sq(a10);
  a23=(a23+a10);
  a20=(a20*a23);
  a20=(a20*a25);
  a6=(a6-a20);
  a6=(a6/a29);
  a16=(a16+a6);
  a1=(a1+a16);
  if (res[0]!=0) res[0][9]=a1;
  a1=(a19*a14);
  a16=(a22*a12);
  a1=(a1-a16);
  a1=(-a1);
  if (res[0]!=0) res[0][10]=a1;
  a1=(a22*a7);
  a14=(a13*a14);
  a1=(a1-a14);
  a1=(-a1);
  if (res[0]!=0) res[0][11]=a1;
  a12=(a13*a12);
  a7=(a19*a7);
  a12=(a12-a7);
  a12=(-a12);
  if (res[0]!=0) res[0][12]=a12;
  a12=0.;
  if (res[0]!=0) res[0][13]=a12;
  if (res[0]!=0) res[0][14]=a12;
  if (res[0]!=0) res[0][15]=a12;
  a12=2.7999999999999997e-02;
  a7=(a15*a32);
  a1=(a5*a31);
  a7=(a7-a1);
  a7=(a12*a7);
  a7=(a13+a7);
  a7=casadi_sq(a7);
  a1=(a17*a32);
  a14=(a21*a31);
  a1=(a1-a14);
  a1=(a12*a1);
  a1=(a19+a1);
  a1=casadi_sq(a1);
  a7=(a7+a1);
  a1=(a18*a32);
  a14=(a0*a31);
  a1=(a1-a14);
  a12=(a12*a1);
  a12=(a22+a12);
  a12=casadi_sq(a12);
  a7=(a7+a12);
  a7=sqrt(a7);
  if (res[0]!=0) res[0][16]=a7;
  a7=-1.2250000000000000e-02;
  a12=(a15*a32);
  a1=(a5*a31);
  a12=(a12-a1);
  a12=(a7*a12);
  a1=2.1217559999999996e-02;
  a14=(a5*a26);
  a16=(a2*a32);
  a14=(a14-a16);
  a14=(a1*a14);
  a12=(a12+a14);
  a12=(a13+a12);
  a12=casadi_sq(a12);
  a14=(a17*a32);
  a16=(a21*a31);
  a14=(a14-a16);
  a14=(a7*a14);
  a16=(a21*a26);
  a6=(a9*a32);
  a16=(a16-a6);
  a16=(a1*a16);
  a14=(a14+a16);
  a14=(a19+a14);
  a14=casadi_sq(a14);
  a12=(a12+a14);
  a14=(a18*a32);
  a16=(a0*a31);
  a14=(a14-a16);
  a14=(a7*a14);
  a16=(a0*a26);
  a6=(a11*a32);
  a16=(a16-a6);
  a1=(a1*a16);
  a14=(a14+a1);
  a14=(a22+a14);
  a14=casadi_sq(a14);
  a12=(a12+a14);
  a12=sqrt(a12);
  if (res[0]!=0) res[0][17]=a12;
  a15=(a15*a32);
  a12=(a5*a31);
  a15=(a15-a12);
  a15=(a7*a15);
  a12=-2.1217559999999996e-02;
  a5=(a5*a26);
  a2=(a2*a32);
  a5=(a5-a2);
  a5=(a12*a5);
  a15=(a15+a5);
  a13=(a13+a15);
  a13=casadi_sq(a13);
  a17=(a17*a32);
  a15=(a21*a31);
  a17=(a17-a15);
  a17=(a7*a17);
  a21=(a21*a26);
  a9=(a9*a32);
  a21=(a21-a9);
  a21=(a12*a21);
  a17=(a17+a21);
  a19=(a19+a17);
  a19=casadi_sq(a19);
  a13=(a13+a19);
  a18=(a18*a32);
  a31=(a0*a31);
  a18=(a18-a31);
  a7=(a7*a18);
  a0=(a0*a26);
  a11=(a11*a32);
  a0=(a0-a11);
  a12=(a12*a0);
  a7=(a7+a12);
  a22=(a22+a7);
  a22=casadi_sq(a22);
  a13=(a13+a22);
  a13=sqrt(a13);
  if (res[0]!=0) res[0][18]=a13;
  return 0;
}

CASADI_SYMBOL_EXPORT int static_solver_position_boundary_expl_ode_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int static_solver_position_boundary_expl_ode_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int static_solver_position_boundary_expl_ode_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void static_solver_position_boundary_expl_ode_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int static_solver_position_boundary_expl_ode_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void static_solver_position_boundary_expl_ode_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void static_solver_position_boundary_expl_ode_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void static_solver_position_boundary_expl_ode_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int static_solver_position_boundary_expl_ode_fun_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int static_solver_position_boundary_expl_ode_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real static_solver_position_boundary_expl_ode_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* static_solver_position_boundary_expl_ode_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* static_solver_position_boundary_expl_ode_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* static_solver_position_boundary_expl_ode_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* static_solver_position_boundary_expl_ode_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int static_solver_position_boundary_expl_ode_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
