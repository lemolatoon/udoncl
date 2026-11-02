// A separate CMake Host target proves that an artifact can combine libraries,
// not only translation units declared by one source target.
int hello_host_source_count(void) { return 3; }
