add_rules("mode.debug", "mode.release")
set_encodings("utf-8")
set_warnings("all")
set_kind("binary")
set_languages("cxx17")

-- 最简单的打印输出
target("draft00")
    add_files("00_simple_print/main.cpp")

-- Epoll边缘触发阻塞测试
target("draft01")
    add_files("01_epoll_edge_blocking/main.cpp")
-- TODO: 可以继续添加更多的draft程序