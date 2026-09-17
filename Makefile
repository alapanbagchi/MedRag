# Repo-root convenience: every target forwards to backend/Makefile.
# Command-line variables (URL=..., URLS=..., FC_QUERY=..., ...) pass through,
# so `make scrape URL=...` works from here and from backend/ alike.

.DEFAULT_GOAL := help

help:
	@$(MAKE) -C backend help

%:
	@$(MAKE) -C backend $@
