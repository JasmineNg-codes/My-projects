set(CMAKE_CXX_COMPILER "/usr/local/software/spack/spack-views/rocky8-icelake-20220710/gcc-11.3.0/gcc-11.3.0/4zpip55j2rww33vhy62jl4eliwynqfru/bin/g++")
set(CMAKE_CXX_COMPILER_ARG1 "")
set(CMAKE_CXX_COMPILER_ID "GNU")
set(CMAKE_CXX_COMPILER_VERSION "11.3.0")
set(CMAKE_CXX_COMPILER_VERSION_INTERNAL "")
set(CMAKE_CXX_COMPILER_WRAPPER "")
set(CMAKE_CXX_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CXX_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CXX_COMPILE_FEATURES "cxx_std_98;cxx_template_template_parameters;cxx_std_11;cxx_alias_templates;cxx_alignas;cxx_alignof;cxx_attributes;cxx_auto_type;cxx_constexpr;cxx_decltype;cxx_decltype_incomplete_return_types;cxx_default_function_template_args;cxx_defaulted_functions;cxx_defaulted_move_initializers;cxx_delegating_constructors;cxx_deleted_functions;cxx_enum_forward_declarations;cxx_explicit_conversions;cxx_extended_friend_declarations;cxx_extern_templates;cxx_final;cxx_func_identifier;cxx_generalized_initializers;cxx_inheriting_constructors;cxx_inline_namespaces;cxx_lambdas;cxx_local_type_template_args;cxx_long_long_type;cxx_noexcept;cxx_nonstatic_member_init;cxx_nullptr;cxx_override;cxx_range_for;cxx_raw_string_literals;cxx_reference_qualified_functions;cxx_right_angle_brackets;cxx_rvalue_references;cxx_sizeof_member;cxx_static_assert;cxx_strong_enums;cxx_thread_local;cxx_trailing_return_types;cxx_unicode_literals;cxx_uniform_initialization;cxx_unrestricted_unions;cxx_user_literals;cxx_variadic_macros;cxx_variadic_templates;cxx_std_14;cxx_aggregate_default_initializers;cxx_attribute_deprecated;cxx_binary_literals;cxx_contextual_conversions;cxx_decltype_auto;cxx_digit_separators;cxx_generic_lambdas;cxx_lambda_init_captures;cxx_relaxed_constexpr;cxx_return_type_deduction;cxx_variable_templates;cxx_std_17;cxx_std_20;cxx_std_23")
set(CMAKE_CXX98_COMPILE_FEATURES "cxx_std_98;cxx_template_template_parameters")
set(CMAKE_CXX11_COMPILE_FEATURES "cxx_std_11;cxx_alias_templates;cxx_alignas;cxx_alignof;cxx_attributes;cxx_auto_type;cxx_constexpr;cxx_decltype;cxx_decltype_incomplete_return_types;cxx_default_function_template_args;cxx_defaulted_functions;cxx_defaulted_move_initializers;cxx_delegating_constructors;cxx_deleted_functions;cxx_enum_forward_declarations;cxx_explicit_conversions;cxx_extended_friend_declarations;cxx_extern_templates;cxx_final;cxx_func_identifier;cxx_generalized_initializers;cxx_inheriting_constructors;cxx_inline_namespaces;cxx_lambdas;cxx_local_type_template_args;cxx_long_long_type;cxx_noexcept;cxx_nonstatic_member_init;cxx_nullptr;cxx_override;cxx_range_for;cxx_raw_string_literals;cxx_reference_qualified_functions;cxx_right_angle_brackets;cxx_rvalue_references;cxx_sizeof_member;cxx_static_assert;cxx_strong_enums;cxx_thread_local;cxx_trailing_return_types;cxx_unicode_literals;cxx_uniform_initialization;cxx_unrestricted_unions;cxx_user_literals;cxx_variadic_macros;cxx_variadic_templates")
set(CMAKE_CXX14_COMPILE_FEATURES "cxx_std_14;cxx_aggregate_default_initializers;cxx_attribute_deprecated;cxx_binary_literals;cxx_contextual_conversions;cxx_decltype_auto;cxx_digit_separators;cxx_generic_lambdas;cxx_lambda_init_captures;cxx_relaxed_constexpr;cxx_return_type_deduction;cxx_variable_templates")
set(CMAKE_CXX17_COMPILE_FEATURES "cxx_std_17")
set(CMAKE_CXX20_COMPILE_FEATURES "cxx_std_20")
set(CMAKE_CXX23_COMPILE_FEATURES "cxx_std_23")

set(CMAKE_CXX_PLATFORM_ID "Linux")
set(CMAKE_CXX_SIMULATE_ID "")
set(CMAKE_CXX_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CXX_SIMULATE_VERSION "")




set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_CXX_COMPILER_AR "/usr/local/software/spack/spack-views/rocky8-icelake-20220710/gcc-11.3.0/gcc-11.3.0/4zpip55j2rww33vhy62jl4eliwynqfru/bin/gcc-ar")
set(CMAKE_RANLIB "/usr/bin/ranlib")
set(CMAKE_CXX_COMPILER_RANLIB "/usr/local/software/spack/spack-views/rocky8-icelake-20220710/gcc-11.3.0/gcc-11.3.0/4zpip55j2rww33vhy62jl4eliwynqfru/bin/gcc-ranlib")
set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_MT "")
set(CMAKE_COMPILER_IS_GNUCXX 1)
set(CMAKE_CXX_COMPILER_LOADED 1)
set(CMAKE_CXX_COMPILER_WORKS TRUE)
set(CMAKE_CXX_ABI_COMPILED TRUE)

set(CMAKE_CXX_COMPILER_ENV_VAR "CXX")

set(CMAKE_CXX_COMPILER_ID_RUN 1)
set(CMAKE_CXX_SOURCE_FILE_EXTENSIONS C;M;c++;cc;cpp;cxx;m;mm;mpp;CPP;ixx;cppm)
set(CMAKE_CXX_IGNORE_EXTENSIONS inl;h;hpp;HPP;H;o;O;obj;OBJ;def;DEF;rc;RC)

foreach (lang C OBJC OBJCXX)
  if (CMAKE_${lang}_COMPILER_ID_RUN)
    foreach(extension IN LISTS CMAKE_${lang}_SOURCE_FILE_EXTENSIONS)
      list(REMOVE_ITEM CMAKE_CXX_SOURCE_FILE_EXTENSIONS ${extension})
    endforeach()
  endif()
endforeach()

set(CMAKE_CXX_LINKER_PREFERENCE 30)
set(CMAKE_CXX_LINKER_PREFERENCE_PROPAGATES 1)

# Save compiler ABI information.
set(CMAKE_CXX_SIZEOF_DATA_PTR "8")
set(CMAKE_CXX_COMPILER_ABI "ELF")
set(CMAKE_CXX_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CXX_LIBRARY_ARCHITECTURE "")

if(CMAKE_CXX_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CXX_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CXX_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CXX_COMPILER_ABI}")
endif()

if(CMAKE_CXX_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX "")
if(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX)
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_CXX_CL_SHOWINCLUDES_PREFIX}")
endif()





set(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "/usr/local/software/spack/spack-views/rocky8-icelake-20220710/ncurses-6.2/intel-2021.6.0/dnilgm7v3ihibdkuirxxkhpsxgtym2jc/include;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/zstd-1.5.2/gcc-11.3.0/xfbmozmj6cva3th7lsb73fs2hgrawq3n/include;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/mpc-1.2.1/gcc-11.3.0/f3azsyi7mk5y3eh7bnitv5sj5vadav5h/include;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/mpfr-4.1.0/gcc-11.3.0/cgv6xkpzaje632spokpqhtmacqmjk2bk/include;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/isl-0.24/gcc-11.3.0/2t2q5axepqsvojk4n4s22pitx3tnmyra/include;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/gmp-6.2.1/gcc-11.3.0/f3azme3k3yopgg3h7cab2v3ctqvvubng/include;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/intel-oneapi-mpi-2021.6.0/intel-2021.6.0/guxuvcpmykplbrr2e3af2yd7njqhau5e/mpi/2021.6.0/include;/usr/local/software/spack/csd3/spack-views/ucx-2024-08-19/ucx-1.15.0/gcc-8.5.0/3odtpik4wnhzdj6fgnc5ujwcmmnx4yjl/include;/usr/local/software/cuda/11.4/include;/usr/local/software/spack/spack-views/._rocky8-icelake-20220710/blj25fni5jxh7kinozeh7b2bcncjmgyd/gcc-11.3.0/gcc-11.3.0/4zpip55j2rww33vhy62jl4eliwynqfru/include/c++/11.3.0;/usr/local/software/spack/spack-views/._rocky8-icelake-20220710/blj25fni5jxh7kinozeh7b2bcncjmgyd/gcc-11.3.0/gcc-11.3.0/4zpip55j2rww33vhy62jl4eliwynqfru/include/c++/11.3.0/x86_64-pc-linux-gnu;/usr/local/software/spack/spack-views/._rocky8-icelake-20220710/blj25fni5jxh7kinozeh7b2bcncjmgyd/gcc-11.3.0/gcc-11.3.0/4zpip55j2rww33vhy62jl4eliwynqfru/include/c++/11.3.0/backward;/usr/local/software/spack/spack-views/._rocky8-icelake-20220710/blj25fni5jxh7kinozeh7b2bcncjmgyd/gcc-11.3.0/gcc-11.3.0/4zpip55j2rww33vhy62jl4eliwynqfru/lib/gcc/x86_64-pc-linux-gnu/11.3.0/include;/usr/local/software/spack/spack-views/._rocky8-icelake-20220710/blj25fni5jxh7kinozeh7b2bcncjmgyd/gcc-11.3.0/gcc-11.3.0/4zpip55j2rww33vhy62jl4eliwynqfru/lib/gcc/x86_64-pc-linux-gnu/11.3.0/include-fixed;/usr/local/include;/usr/local/software/spack/spack-views/._rocky8-icelake-20220710/blj25fni5jxh7kinozeh7b2bcncjmgyd/gcc-11.3.0/gcc-11.3.0/4zpip55j2rww33vhy62jl4eliwynqfru/include;/usr/include")
set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "/usr/local/software/spack/spack-views/._rocky8-icelake-20220710/blj25fni5jxh7kinozeh7b2bcncjmgyd/gcc-11.3.0/gcc-11.3.0/4zpip55j2rww33vhy62jl4eliwynqfru/lib/gcc/x86_64-pc-linux-gnu/11.3.0;/usr/local/software/spack/spack-views/._rocky8-icelake-20220710/blj25fni5jxh7kinozeh7b2bcncjmgyd/gcc-11.3.0/gcc-11.3.0/4zpip55j2rww33vhy62jl4eliwynqfru/lib/gcc;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/gcc-11.3.0/gcc-11.3.0/4zpip55j2rww33vhy62jl4eliwynqfru/lib64;/usr/local/software/cuda/11.4/nvvm/lib64;/usr/local/software/spack/spack-views/._rocky8-icelake-20220710/blj25fni5jxh7kinozeh7b2bcncjmgyd/gcc-11.3.0/gcc-11.3.0/4zpip55j2rww33vhy62jl4eliwynqfru/lib64;/lib64;/usr/lib64;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/ncurses-6.2/intel-2021.6.0/dnilgm7v3ihibdkuirxxkhpsxgtym2jc/lib;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/gcc-11.3.0/gcc-11.3.0/4zpip55j2rww33vhy62jl4eliwynqfru/lib;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/zstd-1.5.2/gcc-11.3.0/xfbmozmj6cva3th7lsb73fs2hgrawq3n/lib;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/mpc-1.2.1/gcc-11.3.0/f3azsyi7mk5y3eh7bnitv5sj5vadav5h/lib;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/mpfr-4.1.0/gcc-11.3.0/cgv6xkpzaje632spokpqhtmacqmjk2bk/lib;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/isl-0.24/gcc-11.3.0/2t2q5axepqsvojk4n4s22pitx3tnmyra/lib;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/gmp-6.2.1/gcc-11.3.0/f3azme3k3yopgg3h7cab2v3ctqvvubng/lib;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/intel-oneapi-mpi-2021.6.0/intel-2021.6.0/guxuvcpmykplbrr2e3af2yd7njqhau5e/mpi/2021.6.0/libfabric/lib;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/intel-oneapi-mpi-2021.6.0/intel-2021.6.0/guxuvcpmykplbrr2e3af2yd7njqhau5e/mpi/2021.6.0/lib/release;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/intel-oneapi-mpi-2021.6.0/intel-2021.6.0/guxuvcpmykplbrr2e3af2yd7njqhau5e/mpi/2021.6.0/lib;/usr/local/software/spack/csd3/spack-views/ucx-2024-08-19/ucx-1.15.0/gcc-8.5.0/3odtpik4wnhzdj6fgnc5ujwcmmnx4yjl/lib;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/intel-oneapi-compilers-2022.1.0/gcc-11.3.0/b6zld2mz7cid27yloxznoidymd7vywwz/compiler/2022.1.0/linux/compiler/lib/intel64_lin;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/intel-oneapi-compilers-2022.1.0/gcc-11.3.0/b6zld2mz7cid27yloxznoidymd7vywwz/compiler/2022.1.0/linux/lib;/usr/local/software/spack/spack-views/rocky8-icelake-20220710/intel-oneapi-compilers-2022.1.0/gcc-11.3.0/b6zld2mz7cid27yloxznoidymd7vywwz/lib;/usr/local/software/cuda/11.4/lib64;/usr/local/software/spack/spack-views/._rocky8-icelake-20220710/blj25fni5jxh7kinozeh7b2bcncjmgyd/gcc-11.3.0/gcc-11.3.0/4zpip55j2rww33vhy62jl4eliwynqfru/lib")
set(CMAKE_CXX_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
