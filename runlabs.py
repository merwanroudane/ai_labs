"""Run named labs from a view file: python runlabs.py views/p15_rnn.py lab1 lab2"""
import io, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.CRITICAL)
for _n in ("streamlit.runtime.caching.cache_data_api",
           "streamlit.runtime.scriptrunner_utils.script_run_context"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)
import lab_test

path, names = sys.argv[1], set(sys.argv[2:])
labs = [t for t in lab_test.extract_labs(path) if not names or t[0] in names]
fails = 0
for key, code, _ln in labs:
    ns = lab_test.base_namespace()
    t0 = time.perf_counter()
    try:
        exec(compile(code, f"<lab:{key}>", "exec"), ns, ns)
        print(f"   OK  {key:<24}{time.perf_counter()-t0:7.1f}s")
    except Exception:
        fails += 1
        print(f"   XX  {key:<24}{time.perf_counter()-t0:7.1f}s")
        print(traceback.format_exc(limit=6))
print(f"== {len(labs)} labs, {fails} failed")
sys.exit(1 if fails else 0)
