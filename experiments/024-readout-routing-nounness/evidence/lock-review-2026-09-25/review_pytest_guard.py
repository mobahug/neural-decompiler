"""pytest plugin: installs the review audit hook (refuses any write under the repository) for the whole session and
reports the refused events at the end; a refused write fails the session."""
import review_guard


def pytest_sessionfinish(session, exitstatus):
    events = review_guard.report_guard()
    print(f"\n[review guard] refused writes: {events['refused_writes']}; refused reads: {events['refused_reads']}")
    if events["refused_writes"] or events["refused_reads"]:
        session.exitstatus = 99
