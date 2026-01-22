"""
Data loading utilities for FactPulse.

This script helps:
1. Load and validate facts data
2. Add new facts to the knowledge base
3. Export/import facts in different formats

Run: python -m scripts.load_data --help
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def validate_fact(fact: Dict[str, Any]) -> List[str]:
    """
    Validate a fact entry.
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    required_fields = ["fact_id", "fact_text", "verdict"]
    for field in required_fields:
        if field not in fact:
            errors.append(f"Missing required field: {field}")
    
    valid_verdicts = ["TRUE", "FALSE", "PARTIALLY_TRUE", "NOT_VERIFIABLE"]
    if "verdict" in fact and fact["verdict"] not in valid_verdicts:
        errors.append(f"Invalid verdict: {fact['verdict']}. Must be one of {valid_verdicts}")
    
    if "sources" in fact:
        if not isinstance(fact["sources"], list):
            errors.append("sources must be a list")
        else:
            for i, source in enumerate(fact["sources"]):
                if "title" not in source:
                    errors.append(f"Source {i} missing 'title' field")
    
    return errors


def load_facts(facts_path: str = "data/known_facts.json") -> List[Dict[str, Any]]:
    """Load facts from JSON file."""
    facts_file = project_root / facts_path
    
    if not facts_file.exists():
        print(f"❌ Facts file not found: {facts_file}")
        return []
    
    with open(facts_file, 'r', encoding='utf-8') as f:
        facts = json.load(f)
    
    return facts


def validate_facts_file(facts_path: str = "data/known_facts.json") -> bool:
    """Validate all facts in a file."""
    facts = load_facts(facts_path)
    
    if not facts:
        return False
    
    print(f"📋 Validating {len(facts)} facts...")
    
    all_valid = True
    fact_ids = set()
    
    for i, fact in enumerate(facts):
        errors = validate_fact(fact)
        
        # Check for duplicate IDs
        if "fact_id" in fact:
            if fact["fact_id"] in fact_ids:
                errors.append(f"Duplicate fact_id: {fact['fact_id']}")
            fact_ids.add(fact["fact_id"])
        
        if errors:
            all_valid = False
            print(f"\n❌ Fact {i + 1} ({fact.get('fact_id', 'unknown')}):")
            for error in errors:
                print(f"   - {error}")
    
    if all_valid:
        print(f"\n✅ All {len(facts)} facts are valid!")
    else:
        print(f"\n❌ Validation failed")
    
    return all_valid


def add_fact(
    fact_text: str,
    verdict: str,
    sources: List[Dict[str, str]] = None,
    facts_path: str = "data/known_facts.json"
) -> str:
    """
    Add a new fact to the knowledge base.
    
    Args:
        fact_text: The factual claim
        verdict: TRUE, FALSE, PARTIALLY_TRUE, or NOT_VERIFIABLE
        sources: List of source dictionaries
        facts_path: Path to facts file
        
    Returns:
        The generated fact_id
    """
    facts = load_facts(facts_path)
    
    # Generate new ID
    existing_ids = {int(f["fact_id"].split("_")[1]) for f in facts if f.get("fact_id", "").startswith("fact_")}
    new_id_num = max(existing_ids, default=0) + 1
    fact_id = f"fact_{new_id_num:03d}"
    
    new_fact = {
        "fact_id": fact_id,
        "fact_text": fact_text,
        "verdict": verdict.upper(),
        "sources": sources or []
    }
    
    # Validate
    errors = validate_fact(new_fact)
    if errors:
        raise ValueError(f"Invalid fact: {errors}")
    
    facts.append(new_fact)
    
    # Save
    facts_file = project_root / facts_path
    with open(facts_file, 'w', encoding='utf-8') as f:
        json.dump(facts, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Added fact: {fact_id}")
    return fact_id


def show_stats(facts_path: str = "data/known_facts.json") -> None:
    """Show statistics about the facts database."""
    facts = load_facts(facts_path)
    
    if not facts:
        print("No facts loaded.")
        return
    
    print("\n📊 Facts Database Statistics")
    print("=" * 40)
    print(f"Total facts: {len(facts)}")
    
    # Verdict breakdown
    verdicts = {}
    for fact in facts:
        v = fact.get("verdict", "UNKNOWN")
        verdicts[v] = verdicts.get(v, 0) + 1
    
    print("\nBy verdict:")
    for verdict, count in sorted(verdicts.items()):
        pct = (count / len(facts)) * 100
        print(f"   {verdict}: {count} ({pct:.1f}%)")
    
    # Source coverage
    with_sources = sum(1 for f in facts if f.get("sources"))
    print(f"\nFacts with sources: {with_sources}/{len(facts)} ({(with_sources/len(facts))*100:.1f}%)")
    
    # Average sources per fact
    total_sources = sum(len(f.get("sources", [])) for f in facts)
    print(f"Average sources per fact: {total_sources/len(facts):.1f}")


def export_facts(
    facts_path: str = "data/known_facts.json",
    output_format: str = "jsonl",
    output_path: str = None
) -> None:
    """Export facts to different formats."""
    facts = load_facts(facts_path)
    
    if not facts:
        return
    
    if output_format == "jsonl":
        output_file = output_path or "data/facts_export.jsonl"
        output_file = project_root / output_file
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for fact in facts:
                f.write(json.dumps(fact, ensure_ascii=False) + "\n")
        
        print(f"💾 Exported to JSONL: {output_file}")
    
    elif output_format == "csv":
        import csv
        output_file = output_path or "data/facts_export.csv"
        output_file = project_root / output_file
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["fact_id", "fact_text", "verdict"])
            writer.writeheader()
            for fact in facts:
                writer.writerow({
                    "fact_id": fact.get("fact_id", ""),
                    "fact_text": fact.get("fact_text", ""),
                    "verdict": fact.get("verdict", "")
                })
        
        print(f"💾 Exported to CSV: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="FactPulse data utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate facts file")
    validate_parser.add_argument("--facts", default="data/known_facts.json", help="Facts file path")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.add_argument("--facts", default="data/known_facts.json", help="Facts file path")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new fact")
    add_parser.add_argument("--text", required=True, help="Fact text")
    add_parser.add_argument("--verdict", required=True, choices=["TRUE", "FALSE", "PARTIALLY_TRUE", "NOT_VERIFIABLE"])
    add_parser.add_argument("--facts", default="data/known_facts.json", help="Facts file path")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export facts")
    export_parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl", help="Output format")
    export_parser.add_argument("--output", help="Output file path")
    export_parser.add_argument("--facts", default="data/known_facts.json", help="Facts file path")
    
    args = parser.parse_args()
    
    if args.command == "validate":
        validate_facts_file(args.facts)
    elif args.command == "stats":
        show_stats(args.facts)
    elif args.command == "add":
        add_fact(args.text, args.verdict, facts_path=args.facts)
    elif args.command == "export":
        export_facts(args.facts, args.format, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
