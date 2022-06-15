file(GLOB tmp *.cpp)
set(CK_CPU_INSTANCE_SRC
   ${CK_CPU_INSTANCE_SRC}
   ${tmp}
   PARENT_SCOPE
)
