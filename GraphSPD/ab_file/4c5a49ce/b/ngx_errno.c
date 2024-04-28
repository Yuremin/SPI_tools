
/*
 * Copyright (C) Igor Sysoev
 * Copyright (C) Nginx, Inc.
 */


#include <ngx_config.h>
#include <ngx_core.h>


static ngx_str_t   ngx_unknown_error = ngx_string("Unknown error");


#if (NGX_HAVE_STRERRORDESC_NP)

/*
 * The strerrordesc_np() function, introduced in glibc 2.32, is
 * async-signal-safe.  This makes it possible to use it directly,
 * without copying error messages.
 */


u_char *
ngx_strerror(ngx_err_t err, u_char *errstr, size_t size)
{
    size_t       len;
    const char  *msg;

    msg = strerrordesc_np(err);

    if (msg == NULL) {
        msg = (char *) ngx_unknown_error.data;
        len = ngx_unknown_error.len;

    } else {
        len = ngx_strlen(msg);
    }

    size = ngx_min(size, len);

    return ngx_cpymem(errstr, msg, size);
}


ngx_int_t
ngx_strerror_init(void)
{
    return NGX_OK;
}


#else

/*
 * The strerror() messages are copied because:
 *
 * 1) strerror() and strerror_r() functions are not Async-Signal-Safe,
 *    therefore, they cannot be used in signal handlers;
 *
 * 2) a direct sys_errlist[] array may be used instead of these functions,
 *    but Linux linker warns about its usage:
 *
 * warning: `sys_errlist' is deprecated; use `strerror' or `strerror_r' instead
 * warning: `sys_nerr' is deprecated; use `strerror' or `strerror_r' instead
 *
 *    causing false bug reports.
 */


static ngx_str_t  *ngx_sys_errlist;

static ngx_err_t   ngx_first_error;
static ngx_err_t   ngx_last_error;








































































































































#endif
