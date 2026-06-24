# Headless XFoil build (for fidelity validation)

`xfoil_validate.py` needs an `xfoil` binary. There is no Homebrew formula and the
stock XFoil 6.99 build links X11 (XQuartz) for its plotting library. Since the
validation runs XFoil graphics-off in batch mode, the X11 dependency was removed
by replacing the plot backend (`Xwin2.c`) with no-op Fortran stubs. The compiled
binary lives at `.venv/bin/xfoil` (on PATH when the venv is active).

## Reproduce (Apple Silicon / macOS)

```bash
brew install gcc                      # provides gfortran
curl -LO https://web.mit.edu/drela/Public/web/xfoil/xfoil6.99.tgz
tar xzf xfoil6.99.tgz && cd Xfoil

# 1) Write Xfoil/plotlib/Xwin_stub.f : empty SUBROUTINEs named GWX* + MSKBITS
#    (one per symbol defined in Xwin2.c) -- they satisfy the linker without X11.

# 2) Build the plot library from the pure-Fortran sources + the stub:
cd plotlib
FLAGS="-O2 -fallow-argument-mismatch -std=legacy"
for f in plt_base plt_font plt_util plt_color set_subs gw_subs ps_subs Xwin_stub; do
  gfortran -c $FLAGS $f.f -o $f.o
done
ar r libPlt.a plt_*.o set_subs.o gw_subs.o ps_subs.o Xwin_stub.o && ranlib libPlt.a

# 3) Build the xfoil binary, no X11 link:
cd ../bin
make xfoil FC=gfortran FFLAGS="$FLAGS" FFLOPT="$FLAGS" \
     PLTOBJ="../plotlib/libPlt.a" PLTLIB=""
# (the final `install` step errors on the author's hardcoded path -- ignore it;
#  the binary `xfoil` is already built in bin/)
```

## Notes
- License: XFoil is GPL (Mark Drela / Harold Youngren). This repo does not
  redistribute it; build it yourself with the steps above.
- The binary is batch-only (no interactive plotting). `xfoil_validate.py` drives
  it via stdin and parses the PACC polar file directly.
