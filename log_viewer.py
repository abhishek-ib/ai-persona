#!/usr/bin/env python3
"""
Log Viewer for AI Persona Bot

This script helps analyze the logged context and responses sent to Gemini.
"""

import json
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any


def parse_log_line(line: str) -> Dict[str, Any]:
    """Parse a log line and extract the JSON data"""
    try:
        # Log format: "TIMESTAMP - LEVEL - MESSAGE"
        # Find the JSON part after "CONTEXT: " or "RESPONSE: "
        if "CONTEXT: " in line:
            json_part = line.split("CONTEXT: ", 1)[1]
            data = json.loads(json_part)
            data['log_type'] = 'context'
            return data
        elif "RESPONSE: " in line:
            json_part = line.split("RESPONSE: ", 1)[1]
            data = json.loads(json_part)
            data['log_type'] = 'response'
            return data
        elif "PINECONE_QUERY: " in line:
            json_part = line.split("PINECONE_QUERY: ", 1)[1]
            data = json.loads(json_part)
            data['log_type'] = 'pinecone_query'
            return data
        elif "PINECONE_USER_CONTEXT: " in line:
            json_part = line.split("PINECONE_USER_CONTEXT: ", 1)[1]
            data = json.loads(json_part)
            data['log_type'] = 'pinecone_user_context'
            return data
        elif "PINECONE_ERROR: " in line or "PINECONE_USER_CONTEXT_ERROR: " in line:
            if "PINECONE_ERROR: " in line:
                json_part = line.split("PINECONE_ERROR: ", 1)[1]
            else:
                json_part = line.split("PINECONE_USER_CONTEXT_ERROR: ", 1)[1]
            data = json.loads(json_part)
            data['log_type'] = 'pinecone_error'
            return data
    except Exception as e:
        print(f"Error parsing line: {e}")
    return None


def load_context_logs(log_file: str = "logs/gemini_context.log") -> List[Dict[str, Any]]:
    """Load and parse context logs"""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return []
    
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            parsed = parse_log_line(line.strip())
            if parsed:
                logs.append(parsed)
    
    return logs


def load_response_logs(log_file: str = "logs/gemini_responses.log") -> List[Dict[str, Any]]:
    """Load and parse response logs"""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return []
    
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            parsed = parse_log_line(line.strip())
            if parsed:
                logs.append(parsed)
    
    return logs


def load_pinecone_logs(log_file: str = "logs/pinecone_queries.log") -> List[Dict[str, Any]]:
    """Load and parse Pinecone query logs"""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return []
    
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            parsed = parse_log_line(line.strip())
            if parsed:
                logs.append(parsed)
    
    return logs


def show_context_summary(logs: List[Dict[str, Any]]):
    """Show summary of context logs"""
    print("📊 Context Logs Summary")
    print("=" * 50)
    
    if not logs:
        print("No context logs found")
        return
    
    print(f"Total queries: {len(logs)}")
    
    # User statistics
    users = {}
    for log in logs:
        user = log.get('target_user', 'Unknown')
        if user not in users:
            users[user] = 0
        users[user] += 1
    
    print(f"\nQueries by user:")
    for user, count in sorted(users.items(), key=lambda x: x[1], reverse=True):
        print(f"  {user}: {count}")
    
    # Average similarity scores
    similarity_scores = []
    for log in logs:
        for msg in log.get('similar_messages', []):
            score = msg.get('similarity_score', 0)
            if score > 0:
                similarity_scores.append(score)
    
    if similarity_scores:
        avg_score = sum(similarity_scores) / len(similarity_scores)
        max_score = max(similarity_scores)
        min_score = min(similarity_scores)
        print(f"\nSimilarity scores:")
        print(f"  Average: {avg_score:.3f}")
        print(f"  Max: {max_score:.3f}")
        print(f"  Min: {min_score:.3f}")


def show_recent_contexts(logs: List[Dict[str, Any]], limit: int = 5):
    """Show recent context logs"""
    print(f"\n📝 Recent Context Logs (last {limit})")
    print("=" * 50)
    
    if not logs:
        print("No context logs found")
        return
    
    # Sort by timestamp and take most recent
    sorted_logs = sorted(logs, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    for i, log in enumerate(sorted_logs[:limit], 1):
        timestamp = log.get('timestamp', 'Unknown')
        if timestamp != 'Unknown':
            try:
                dt = datetime.fromisoformat(timestamp)
                timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass
        
        query = log.get('query', 'Unknown')[:80]
        user = log.get('target_user', 'Unknown')
        similar_count = log.get('similar_messages_count', 0)
        
        print(f"\n{i}. [{timestamp}] {user}")
        print(f"   Query: {query}...")
        print(f"   Similar messages used: {similar_count}")
        
        # Show top similar messages
        similar_msgs = log.get('similar_messages', [])[:3]
        if similar_msgs:
            print("   Top similar messages:")
            for j, msg in enumerate(similar_msgs, 1):
                score = msg.get('similarity_score', 0)
                msg_user = msg.get('user_name', 'Unknown')
                content = msg.get('content', '')[:60]
                print(f"     {j}. [{score:.3f}] {msg_user}: {content}...")


def show_query_details(logs: List[Dict[str, Any]], query_text: str):
    """Show detailed information for queries containing specific text"""
    print(f"🔍 Queries containing '{query_text}'")
    print("=" * 50)
    
    matching_logs = [log for log in logs if query_text.lower() in log.get('query', '').lower()]
    
    if not matching_logs:
        print("No matching queries found")
        return
    
    for i, log in enumerate(matching_logs, 1):
        timestamp = log.get('timestamp', 'Unknown')
        if timestamp != 'Unknown':
            try:
                dt = datetime.fromisoformat(timestamp)
                timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass
        
        print(f"\n{i}. [{timestamp}]")
        print(f"   Query: {log.get('query', 'Unknown')}")
        print(f"   Target user: {log.get('target_user', 'Unknown')}")
        print(f"   Similar messages: {log.get('similar_messages_count', 0)}")
        print(f"   User context messages: {log.get('user_recent_messages_count', 0)}")
        
        # Show all similar messages for this query
        similar_msgs = log.get('similar_messages', [])
        if similar_msgs:
            print("   Similar messages sent to Gemini:")
            for j, msg in enumerate(similar_msgs, 1):
                score = msg.get('similarity_score', 0)
                msg_user = msg.get('user_name', 'Unknown')
                content = msg.get('content', '')
                msg_time = msg.get('datetime', 'Unknown')
                print(f"     {j}. [{score:.3f}] {msg_user} ({msg_time})")
                print(f"        {content[:150]}...")


def analyze_user_context(logs: List[Dict[str, Any]], user_name: str):
    """Analyze context for a specific user"""
    print(f"👤 Context Analysis for {user_name}")
    print("=" * 50)
    
    user_logs = [log for log in logs if log.get('target_user', '').lower() == user_name.lower()]
    
    if not user_logs:
        print(f"No logs found for user: {user_name}")
        return
    
    print(f"Total queries for {user_name}: {len(user_logs)}")
    
    # Analyze what context is being used
    context_sources = {}
    for log in user_logs:
        for msg in log.get('similar_messages', []):
            source_user = msg.get('user_name', 'Unknown')
            if source_user not in context_sources:
                context_sources[source_user] = 0
            context_sources[source_user] += 1
    
    print(f"\nContext sources (who's messages are used as context):")
    for source, count in sorted(context_sources.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count} messages")
    
    # Show recent queries
    print(f"\nRecent queries for {user_name}:")
    sorted_logs = sorted(user_logs, key=lambda x: x.get('timestamp', ''), reverse=True)
    for i, log in enumerate(sorted_logs[:5], 1):
        query = log.get('query', 'Unknown')[:80]
        similar_count = log.get('similar_messages_count', 0)
        print(f"  {i}. {query}... (used {similar_count} similar messages)")


def show_pinecone_summary(logs: List[Dict[str, Any]]):
    """Show summary of Pinecone query logs"""
    print("🔍 Pinecone Query Logs Summary")
    print("=" * 50)
    
    if not logs:
        print("No Pinecone logs found")
        return
    
    # Separate by log type
    query_logs = [log for log in logs if log.get('log_type') == 'pinecone_query']
    context_logs = [log for log in logs if log.get('log_type') == 'pinecone_user_context']
    error_logs = [log for log in logs if log.get('log_type') == 'pinecone_error']
    
    print(f"Total Pinecone operations: {len(logs)}")
    print(f"  - Similarity queries: {len(query_logs)}")
    print(f"  - User context queries: {len(context_logs)}")
    print(f"  - Errors: {len(error_logs)}")
    
    if query_logs:
        # Analyze similarity scores
        all_scores = []
        for log in query_logs:
            for result in log.get('results', []):
                score = result.get('similarity_score', 0)
                if score > 0:
                    all_scores.append(score)
        
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            max_score = max(all_scores)
            min_score = min(all_scores)
            print(f"\nSimilarity scores:")
            print(f"  Average: {avg_score:.3f}")
            print(f"  Max: {max_score:.3f}")
            print(f"  Min: {min_score:.3f}")
        
        # Most common queries
        query_counts = {}
        for log in query_logs:
            query = log.get('query', '')[:50] + '...'
            query_counts[query] = query_counts.get(query, 0) + 1
        
        if query_counts:
            print(f"\nMost common queries:")
            for query, count in sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {count}x: {query}")


def show_pinecone_queries(logs: List[Dict[str, Any]], limit: int = 5):
    """Show recent Pinecone queries with results"""
    print(f"\n🔍 Recent Pinecone Queries (last {limit})")
    print("=" * 50)
    
    query_logs = [log for log in logs if log.get('log_type') == 'pinecone_query']
    if not query_logs:
        print("No Pinecone query logs found")
        return
    
    # Sort by timestamp and take most recent
    sorted_logs = sorted(query_logs, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    for i, log in enumerate(sorted_logs[:limit], 1):
        timestamp = log.get('timestamp', 'Unknown')
        if timestamp != 'Unknown':
            try:
                dt = datetime.fromisoformat(timestamp)
                timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass
        
        query = log.get('query', 'Unknown')
        results_count = log.get('results_count', 0)
        user_filter = log.get('user_filter')
        
        print(f"\n{i}. [{timestamp}]")
        print(f"   Query: {query}")
        if user_filter:
            print(f"   Filtered by user: {user_filter}")
        print(f"   Results returned: {results_count}")
        
        # Show top results
        results = log.get('results', [])[:3]
        if results:
            print("   Top results from Pinecone:")
            for j, result in enumerate(results, 1):
                score = result.get('similarity_score', 0)
                user = result.get('user_name', 'Unknown')
                content = result.get('content', '')
                print(f"     {j}. [{score:.3f}] {user}: {content}")


def analyze_pinecone_performance(logs: List[Dict[str, Any]]):
    """Analyze Pinecone query performance and patterns"""
    print("📈 Pinecone Performance Analysis")
    print("=" * 50)
    
    query_logs = [log for log in logs if log.get('log_type') == 'pinecone_query']
    if not query_logs:
        print("No query logs to analyze")
        return
    
    # Analyze result counts
    result_counts = [log.get('results_count', 0) for log in query_logs]
    if result_counts:
        avg_results = sum(result_counts) / len(result_counts)
        max_results = max(result_counts)
        min_results = min(result_counts)
        
        print(f"Results per query:")
        print(f"  Average: {avg_results:.1f}")
        print(f"  Max: {max_results}")
        print(f"  Min: {min_results}")
    
    # Analyze queries by user filter
    filtered_queries = [log for log in query_logs if log.get('user_filter')]
    unfiltered_queries = [log for log in query_logs if not log.get('user_filter')]
    
    print(f"\nQuery types:")
    print(f"  Filtered by user: {len(filtered_queries)}")
    print(f"  Unfiltered (global search): {len(unfiltered_queries)}")
    
    # Show users most queried for
    if filtered_queries:
        user_query_counts = {}
        for log in filtered_queries:
            user = log.get('user_filter', 'Unknown')
            user_query_counts[user] = user_query_counts.get(user, 0) + 1
        
        print(f"\nMost queried users:")
        for user, count in sorted(user_query_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {user}: {count} queries")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AI Persona Bot Log Viewer")
    parser.add_argument("--summary", action="store_true", help="Show context summary")
    parser.add_argument("--recent", type=int, default=5, help="Show recent contexts (default: 5)")
    parser.add_argument("--query", type=str, help="Search for specific query text")
    parser.add_argument("--user", type=str, help="Analyze context for specific user")
    parser.add_argument("--pinecone", action="store_true", help="Show Pinecone query analysis")
    parser.add_argument("--pinecone-queries", type=int, help="Show recent Pinecone queries")
    parser.add_argument("--pinecone-performance", action="store_true", help="Analyze Pinecone performance")
    parser.add_argument("--context-log", default="logs/gemini_context.log", help="Context log file path")
    parser.add_argument("--response-log", default="logs/gemini_responses.log", help="Response log file path")
    parser.add_argument("--pinecone-log", default="logs/pinecone_queries.log", help="Pinecone log file path")
    
    args = parser.parse_args()
    
    # Load logs
    print("📂 Loading logs...")
    context_logs = load_context_logs(args.context_log)
    response_logs = load_response_logs(args.response_log)
    pinecone_logs = load_pinecone_logs(args.pinecone_log)
    
    if not context_logs and not response_logs and not pinecone_logs:
        print("No logs found. Run the bot with logging enabled to generate logs.")
        return
    
    if args.pinecone:
        show_pinecone_summary(pinecone_logs)
    elif args.pinecone_queries:
        show_pinecone_queries(pinecone_logs, args.pinecone_queries)
    elif args.pinecone_performance:
        analyze_pinecone_performance(pinecone_logs)
    elif args.summary:
        show_context_summary(context_logs)
    elif args.query:
        show_query_details(context_logs, args.query)
    elif args.user:
        analyze_user_context(context_logs, args.user)
    else:
        show_recent_contexts(context_logs, args.recent)


if __name__ == "__main__":
    main()
