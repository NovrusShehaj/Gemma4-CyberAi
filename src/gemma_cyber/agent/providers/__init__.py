"""Chat providers: adapters that turn a vendor stream into `StreamEvent`s.

Providers own protocol quirks and nothing else. They do not decide permissions,
do not touch the filesystem, and do not know what a tool does.
"""
