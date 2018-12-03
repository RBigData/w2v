AC_INIT(DESCRIPTION)

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi
CC=`"${R_HOME}/bin/R" CMD config CC`
R_CPPFLAGS=`"${R_HOME}/bin/R" CMD config --cppflags`


SPMD_CPPFLAGS=`"${R_HOME}/bin/Rscript" -e "source('./R/conf/get_conf.r');get.conf('PKG_CPPFLAGS','"${R_ARCH}"')"`
SPMD_LDFLAGS=`"${R_HOME}/bin/Rscript" -e "source('./R/conf/get_conf.r');get.conf('PKG_LIBS','"${R_ARCH}"')"`


# Report
echo " "
echo "******************* Results of w2v package configure *******************"
echo " "
echo "* MPI report:"
echo "    >> SPMD_CPPFLAGS = ${SPMD_CPPFLAGS}"
echo "    >> SPMD_LDFLAGS = ${SPMD_LDFLAGS}"
echo "*************************************************************************"
echo " "


AC_SUBST(SPMD_CPPFLAGS)
AC_SUBST(SPMD_LDFLAGS)

AC_OUTPUT(src/Makevars)
