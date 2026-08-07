
#ifndef DDS_EXPORT_H
#define DDS_EXPORT_H

#ifdef DDS_STATIC_DEFINE
#  define DDS_EXPORT
#  define DDS_NO_EXPORT
#else
#  ifndef DDS_EXPORT
#    ifdef ddsc_EXPORTS
        /* We are building this library */
#      define DDS_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define DDS_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef DDS_NO_EXPORT
#    define DDS_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef DDS_DEPRECATED
#  define DDS_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef DDS_DEPRECATED_EXPORT
#  define DDS_DEPRECATED_EXPORT DDS_EXPORT DDS_DEPRECATED
#endif

#ifndef DDS_DEPRECATED_NO_EXPORT
#  define DDS_DEPRECATED_NO_EXPORT DDS_NO_EXPORT DDS_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef DDS_NO_DEPRECATED
#    define DDS_NO_DEPRECATED
#  endif
#endif

#ifndef DDS_INLINE_EXPORT
#  if __MINGW32__ && (!defined(__clang__) || !defined(ddsc_EXPORTS))
#    define DDS_INLINE_EXPORT
#  else
#    define DDS_INLINE_EXPORT DDS_EXPORT
#  endif
#endif

#endif /* DDS_EXPORT_H */
