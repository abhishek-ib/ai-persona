"""
Chat Session Management for AI Persona Bot

This module handles conversation continuity, context tracking, and query expansion
to maintain coherent multi-turn conversations.
"""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json


class ChatSession:
    """Manages a single chat session with context tracking"""
    
    def __init__(self, user_id: str, session_timeout: int = 1800):  # 30 minutes default
        self.user_id = user_id
        self.session_id = f"{user_id}_{int(time.time())}"
        self.session_timeout = session_timeout
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # Conversation history
        self.conversation_history: List[Dict[str, Any]] = []
        self.context_entities: Dict[str, Any] = {}  # Track mentioned entities
        
    def is_expired(self) -> bool:
        """Check if session has expired"""
        return (datetime.now() - self.last_activity).seconds > self.session_timeout
    
    def add_exchange(self, query: str, response: str, context: Dict[str, Any]):
        """Add a query-response exchange to the session"""
        self.last_activity = datetime.now()
        
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'context_summary': self._extract_context_summary(context)
        }
        
        self.conversation_history.append(exchange)
        
        # Update context entities
        self._update_entities(query, context)
        
        # Keep only recent history (last 10 exchanges)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def expand_query(self, query: str) -> str:
        """Expand query with context from conversation history"""
        expanded_query = query
        
        # Check if query has pronouns or ambiguous references
        pronouns = ['he', 'she', 'it', 'they', 'them', 'his', 'her', 'their', 'this', 'that']
        has_pronouns = any(pronoun in query.lower().split() for pronoun in pronouns)
        
        if has_pronouns and self.conversation_history:
            # Get context from recent exchanges
            context_info = self._get_contextual_info()
            if context_info:
                expanded_query = f"{query} (Context: {context_info})"
        
        return expanded_query
    
    def get_conversation_context(self) -> str:
        """Get formatted conversation context for prompts"""
        if not self.conversation_history:
            return ""
        
        context_lines = ["Recent conversation context:"]
        for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
            timestamp = exchange['timestamp'][:16]  # YYYY-MM-DD HH:MM
            context_lines.append(f"[{timestamp}] User: {exchange['query']}")
            context_lines.append(f"[{timestamp}] Assistant: {exchange['response']}")
        
        return "\n".join(context_lines)
    
    def _extract_context_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key information from context (handles both old and new JSON formats)"""
        summary = {}
        
        # Handle new JSON format
        if 'similar_conversations' in context and isinstance(context['similar_conversations'], list):
            users_mentioned = set()
            topics = []
            
            for conv in context['similar_conversations'][:3]:
                # Extract participants
                if 'participants' in conv:
                    users_mentioned.update(conv['participants'])
                
                # Extract topics from messages
                messages = conv.get('messages', [])
                for msg in messages:
                    message_text = msg.get('message', '')
                    words = message_text.lower().split()
                    topics.extend([w for w in words if len(w) > 5 and w.isalpha()])
            
            summary['users_mentioned'] = list(users_mentioned)
            summary['topics'] = list(set(topics))[:10]
        
        # Handle legacy format (for backward compatibility)
        elif 'similar_conversations' in context:
            # This might be a count or other format
            summary['conversation_count'] = context.get('similar_conversations', 0)
        
        return summary
    
    def _update_entities(self, query: str, context: Dict[str, Any]):
        """Update tracked entities from query and context"""
        # Extract names mentioned in query
        words = query.split()
        potential_names = [w for w in words if w[0].isupper() and w.isalpha()]
        
        for name in potential_names:
            self.context_entities[name.lower()] = {
                'name': name,
                'last_mentioned': datetime.now().isoformat(),
                'context': 'query'
            }
        
        # Extract entities from context
        if 'similar_conversations' in context:
            for conv in context['similar_conversations'][:3]:
                if 'participants' in conv:
                    for participant in conv['participants']:
                        self.context_entities[participant.lower()] = {
                            'name': participant,
                            'last_mentioned': datetime.now().isoformat(),
                            'context': 'conversation'
                        }
    
    def _get_contextual_info(self) -> str:
        """Get contextual information to help resolve pronouns"""
        context_parts = []
        
        # Recent entities mentioned
        recent_entities = []
        for entity_key, entity_info in self.context_entities.items():
            last_mentioned = datetime.fromisoformat(entity_info['last_mentioned'])
            if (datetime.now() - last_mentioned).seconds < 300:  # Within 5 minutes
                recent_entities.append(entity_info['name'])
        
        if recent_entities:
            context_parts.append(f"recently mentioned: {', '.join(recent_entities[:3])}")
        
        # Recent conversation topics
        if self.conversation_history:
            last_exchange = self.conversation_history[-1]
            last_query = last_exchange['query']
            # Extract key terms from last query
            key_terms = [w for w in last_query.split() if len(w) > 4 and w.isalpha()]
            if key_terms:
                context_parts.append(f"previous topic: {' '.join(key_terms[:3])}")
        
        return "; ".join(context_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization"""
        return {
            'user_id': self.user_id,
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'conversation_history': self.conversation_history,
            'context_entities': self.context_entities
        }


class ChatSessionManager:
    """Manages multiple chat sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
    
    def get_or_create_session(self, user_id: str) -> ChatSession:
        """Get existing session or create new one"""
        # Clean up expired sessions
        self._cleanup_expired_sessions()
        
        if user_id not in self.sessions or self.sessions[user_id].is_expired():
            self.sessions[user_id] = ChatSession(user_id)
        
        return self.sessions[user_id]
    
    def add_exchange(self, user_id: str, query: str, response: str, context: Dict[str, Any]):
        """Add exchange to user's session"""
        session = self.get_or_create_session(user_id)
        session.add_exchange(query, response, context)
    
    def expand_query(self, user_id: str, query: str) -> str:
        """Expand query with session context"""
        session = self.get_or_create_session(user_id)
        return session.expand_query(query)
    
    def get_conversation_context(self, user_id: str) -> str:
        """Get conversation context for user"""
        if user_id in self.sessions and not self.sessions[user_id].is_expired():
            return self.sessions[user_id].get_conversation_context()
        return ""
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        expired_users = [
            user_id for user_id, session in self.sessions.items()
            if session.is_expired()
        ]
        
        for user_id in expired_users:
            del self.sessions[user_id]
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about active sessions"""
        self._cleanup_expired_sessions()
        
        return {
            'active_sessions': len(self.sessions),
            'total_exchanges': sum(
                len(session.conversation_history) 
                for session in self.sessions.values()
            ),
            'users_with_sessions': list(self.sessions.keys())
        }


# Global session manager instance
session_manager = ChatSessionManager()
