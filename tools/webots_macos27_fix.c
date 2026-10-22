/*
 * webots_macos27_fix.c
 *
 * Workaround for Webots R2025a on macOS 27: Webots refuses to start with
 * "failed to open TCP server in the port range [1234-1244]" although no port is used.
 * Bug report: https://github.com/cyberbotics/webots/issues/6976
 * Root cause (from upstream PR https://github.com/cyberbotics/webots/pull/7032):
 * on macOS 27 a refused non-blocking connect() makes the follow-up connect() return
 * EISCONN, which Qt treats as "connected", so Webots thinks every port is taken.
 *
 * This library intercepts connect(): if it reports EISCONN but the socket has no
 * real peer (getpeername fails), it reports ECONNREFUSED instead. Nothing else changes.
 * It is loaded only when Webots is started through run_webots.sh (DYLD_INSERT_LIBRARIES);
 * the Webots installation itself is not modified.
 */
#include <errno.h>
#include <sys/socket.h>

#define DYLD_INTERPOSE(_replacement, _replacee)                                         \
  __attribute__((used)) static struct {                                                 \
    const void *replacement;                                                            \
    const void *replacee;                                                               \
  } _interpose_##_replacee __attribute__((section("__DATA,__interpose"))) = {          \
    (const void *)(unsigned long)&_replacement, (const void *)(unsigned long)&_replacee \
  };

static int fixed_connect(int fd, const struct sockaddr *addr, socklen_t len) {
  int r = connect(fd, addr, len); /* inside the interposing library this is the real connect */
  if (r == -1 && errno == EISCONN) {
    struct sockaddr_storage peer;
    socklen_t peer_len = sizeof(peer);
    if (getpeername(fd, (struct sockaddr *)&peer, &peer_len) != 0) {
      errno = ECONNREFUSED; /* not really connected: nobody is listening */
      return -1;
    }
  }
  return r;
}
DYLD_INTERPOSE(fixed_connect, connect)
/* Reference: the code above was generated with Claude (Anthropic LLM), 2026-10-18,
   based on the analysis in https://github.com/cyberbotics/webots/pull/7032 */
