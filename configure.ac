AC_INIT(DESCRIPTION)

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi
CC=`"${R_HOME}/bin/R" CMD config CC`
R_CPPFLAGS=`"${R_HOME}/bin/R" CMD config --cppflags`
LDFLAGS=`"${R_HOME}/bin/R" CMD config LDFLAGS`
BLAS_LIBS=`${R_HOME}/bin/R CMD config BLAS_LIBS`

AC_CHECK_PROG([echo_ok], [echo], [yes], [no], ,)
AC_CHECK_PROG([grep_ok], [grep], [yes], [no], ,)



### MPI
SPMD_CPPFLAGS=`"${R_HOME}/bin/Rscript" -e "source('./R/conf/get_conf.r');get.conf('PKG_CPPFLAGS','"${R_ARCH}"')"`
SPMD_LDFLAGS=`"${R_HOME}/bin/Rscript" -e "source('./R/conf/get_conf.r');get.conf('PKG_LIBS','"${R_ARCH}"')"`



### BLAS
LDFLAGS_OLD="${LDFLAGS}"
LDFLAGS="${LDFLAGS} ${BLAS_LIBS}"

AC_CHECK_LIB(mkl, [mkl_set_num_threads], [ISLIBMKL=yes], [ISLIBMKL=no])
AC_CHECK_LIB(openblas, [openblas_set_num_threads], [ISLIBOPENBLAS=yes], [ISLIBOPENBLAS=no])

if test "X${ISLIBMKL}" = "Xyes"; then
  BLAS_BACKEND="BLAS_USE_MKL"
elif test "X${ISLIBOPENBLAS}" = "Xyes"; then
  BLAS_BACKEND="BLAS_USE_OPENBLAS"
else
  BLAS_BACKEND="BLAS_UNKNOWN"
fi

LDFLAGS=${LDFLAGS_OLD}



# Report
echo " "
echo "******************* Results of w2v package configure *******************"
echo " "
echo "* MPI report:"
echo "    >> SPMD_CPPFLAGS = ${SPMD_CPPFLAGS}"
echo "    >> SPMD_LDFLAGS = ${SPMD_LDFLAGS}"
echo "* BLAS report:"
echo "    >> BLAS_BACKEND = ${BLAS_BACKEND}"
echo "*************************************************************************"
echo " "


AC_SUBST(SPMD_CPPFLAGS)
AC_SUBST(SPMD_LDFLAGS)
AC_OUTPUT(src/Makevars)

AC_SUBST(BLAS_BACKEND)
AC_OUTPUT(src/pWord2Vec/blas_backend.h)
