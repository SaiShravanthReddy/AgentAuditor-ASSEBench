"""Optional stage, run manually after 'infer' when AgentAuditor/temp/<dataset>/failed.json is
non-empty: retries only the items infer.py's LLM calls failed on (API errors exhausting
MAX_RETRIES), instead of infer.py's own behavior of reprocessing the entire dataset from scratch on
every run. Merges any newly-successful items into output-k3.json and shrinks failed.json to
whatever's still failing, so repeated retry passes converge instead of wasting API calls
re-requesting already-successful items every time.

Not wired into the main 'infer'/'demo'/etc. pipeline stages - failed.json only exists after a
partial infer failure, so this is meant to be run on demand:
    python -m AgentAuditor <dataset> infer_retry
"""
import json
import os
from typing import Any, Dict, List

from .infer import GPTConfig, LLMHandler


def _parse_llm_output(item: Dict[str, Any], llm_output: str) -> Dict[str, Any]:
    """Mirrors infer.py's process_json_file inline parsing exactly, so a retried item that
    succeeds ends up in the identical shape as one infer.py itself would have produced."""
    new_item = item.copy()
    llm_output = llm_output.strip()
    if llm_output.startswith('"'):
        llm_output = llm_output[1:]
    if llm_output.endswith('"'):
        llm_output = llm_output[:-1]
    try:
        new_item['output'] = json.loads(llm_output)
    except json.JSONDecodeError as e:
        new_item['output'] = llm_output
        new_item['output_parse_error'] = str(e)
    return new_item


def append_recovered_item(output_file: str, new_item: Dict[str, Any]) -> int:
    """Re-reads output_file fresh immediately before writing, rather than trusting a long-lived
    in-memory accumulator held across the whole retry loop - makes each write self-contained.

    Root-caused a real production bug this way: the old version loaded output_data once at the
    start of retry_failed_main and appended to that single in-memory list for the entire loop,
    trusting it was authoritative. On real HPG runs, the majority of items that genuinely succeeded
    (confirmed via the job's own log output) still ended up missing from the final output-k3.json -
    e.g. 31/33 "recovered" items on one real cnfinbench-pooled run, reproduced again on a second,
    independent run. The exact mechanism was never conclusively identified (code review of this
    function, fix1_main, and the timer.py wrapper found no bug in isolation - each re-read here
    makes that moot rather than relying on fully explaining it), but re-reading fresh before every
    single write and deduping by id removes the possibility of a long-lived in-memory snapshot
    drifting from disk, whatever the cause. Returns the item count after this write.
    """
    with open(output_file, 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    output_data = [item for item in output_data if item.get('id') != new_item.get('id')]
    output_data.append(new_item)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    return len(output_data)


def retry_failed_main(dataset: str) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, f"../temp/{dataset}/output-k3.json")
    failed_items_file = os.path.join(script_dir, f"../temp/{dataset}/failed.json")

    if not os.path.exists(failed_items_file):
        print(f"No {failed_items_file} - nothing to retry.")
        return

    with open(failed_items_file, 'r', encoding='utf-8') as f:
        failed_items = json.load(f)
    if not failed_items:
        print("failed.json is empty - nothing to retry.")
        return

    print(f"Retrying {len(failed_items)} previously-failed item(s)...")
    config = GPTConfig()
    llm_handler = LLMHandler(config)

    still_failed: List[Dict[str, Any]] = []
    recovered = 0
    final_count = None
    for idx, raw_item in enumerate(failed_items):
        item = {k: v for k, v in raw_item.items() if k != 'failure_reason'}
        llm_output = llm_handler.call_llm_api(item['combined_prompt'], item['id'])
        if llm_output is None:
            failed_item = item.copy()
            failed_item['failure_reason'] = "API call failed after maximum retries (retry pass)"
            still_failed.append(failed_item)
        else:
            final_count = append_recovered_item(output_file, _parse_llm_output(item, llm_output))
            recovered += 1

        # Save progress after each item, same crash-safety rationale as infer.py: if this gets
        # interrupted partway, failed.json still accurately reflects what's left to retry next time.
        with open(failed_items_file, 'w', encoding='utf-8') as f:
            json.dump(still_failed + failed_items[idx + 1:], f, indent=2, ensure_ascii=False)

    if final_count is None:
        with open(output_file, 'r', encoding='utf-8') as f:
            final_count = len(json.load(f))
    print(f"Recovered {recovered}/{len(failed_items)} item(s). "
          f"{len(still_failed)} still failing - {output_file} now has {final_count} records.")
