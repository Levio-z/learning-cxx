#include "test.h"
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: xmake run draft <draft number>" << std::endl;
        return EXIT_FAILURE;
    }
    int num;
    if (1 != std::sscanf(argv[1], "%d", &num)) {
        std::cerr << "Invalid draft number: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    };
    Log{Console{}} >> num;  // 使用新的操作符 >> 来运行draft
    return EXIT_SUCCESS;
}