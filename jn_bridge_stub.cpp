// jn_bridge_stub.cpp -- linked when the build found no j_neuro checkout (CMakeLists.txt). Every
// pool is refused at creation, so --jn-stocknn and the JN race entries fail at startup with a
// message; nothing past jn_pool_create is reachable.
#include "jn_bridge.h"

JNPool* jn_pool_create(int, int, int, int, const std::string&, uint64_t, std::string& err) {
    err = "this trainer was built without j_neuro (set -DJNEURO_DIR=<checkout> and rebuild)";
    return nullptr;
}
void    jn_pool_free(JNPool*) {}
int     jn_n_slots(const JNPool*) { return 0; }
void    jn_push_row(JNPool*, const float*) {}
void    jn_replace_row(JNPool*, const float*) {}
void    jn_clear_rows(JNPool*) {}
bool    jn_forward(JNPool*, int, float*) { return false; }
void    jn_mutate(JNPool*, int, int, uint64_t, uint64_t) {}
void    jn_permute(JNPool*, const int*, int) {}
void    jn_inject(JNPool*, int, double, uint64_t) {}
JNStats jn_stats(const JNPool*) { return {}; }
bool    jn_save(const JNPool*, const std::string&, const std::string&) { return false; }
bool    jn_load(JNPool*, const std::string&, const std::string&) { return false; }
bool    jn_save_slot(const JNPool*, int, const std::string&) { return false; }
