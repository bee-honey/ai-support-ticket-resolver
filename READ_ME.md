Support Ticket Resolver

A RAG Assistant for Grounded Ticket Resolution and Triage

A RAG-powered support assistant that retrieves similar historical tickets and relevant support documentation to generate grounded, explainable resolution suggestions.

Tech Stack

Python · FastAPI · Streamlit · LangChain · ChromaDB · SQLite · SQLAlchemy · Pydantic · LLMs · RAG

⸻

1. Overview

Support teams often spend significant time investigating issues that have already been solved. A new ticket may describe the same underlying problem differently, making previous resolutions difficult to discover.

Support documentation can also become outdated or conflict with newer resolutions.

The Support Ticket Resolver uses Retrieval-Augmented Generation (RAG) to search:

* Previously resolved support tickets
* Support documentation
* Troubleshooting guides
* Runbooks and FAQs

The system uses retrieved information to generate a grounded suggested resolution for a new support ticket.

Rather than automatically taking actions, the system acts as an AI assistant for support engineers, providing suggested resolutions along with the evidence used to generate them.

⸻

2. Core Goals

For a newly created support ticket, the system should:

1. Store and manage the ticket.
2. Search previously resolved tickets for similar issues.
3. Search relevant support documentation.
4. Retrieve the most relevant context.
5. Generate a suggested resolution using an LLM.
6. Show the evidence and sources used to generate the answer.
7. Identify similar historical tickets.
8. Warn when retrieved documentation may be outdated or conflict with newer information.
9. Allow the user to provide feedback on the generated resolution.

Key principle: The emphasis is on grounded AI responses rather than building a generic chatbot.

⸻

3. Scope

Core Capstone

The initial implementation will focus on:

* Python
* Streamlit
* FastAPI
* Uvicorn
* SQLite
* SQLAlchemy
* Pydantic
* LangChain
* ChromaDB
* Embeddings
* LLM integration
* Retrieval-Augmented Generation (RAG)
* Source attribution
* Similar-ticket retrieval
* Conflict and staleness detection
* Evaluation

Stretch Goals

The following are intentionally not required for the first implementation:

* LangGraph
* AI agents
* Multi-agent systems
* Model Context Protocol (MCP)
* Autonomous tool execution
* Complex workflow orchestration

These should only be considered after the core RAG application is complete and demo-ready.

⸻

4. High-Level Architecture

                         Support Ticket Resolver
┌────────────────────────────────────────────────────────────┐
│                        Streamlit                           │
│                                                            │
│ Create Ticket │ Ticket List │ Resolution │ Feedback        │
└─────────────────────────────┬──────────────────────────────┘
                              │
                           HTTP/JSON
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                         FastAPI                            │
│                                                            │
│ /tickets │ /resolve │ /similar │ /feedback                 │
└─────────────────────────────┬──────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                   Application Services                     │
│                                                            │
│ TicketService │ ResolutionService │ RetrievalService       │
└───────────────────┬───────────────────────────┬────────────┘
                    │                           │
                    ▼                           ▼
               SQLite                    LangChain RAG
                                              │
                                              ▼
                                          ChromaDB
                                         /        \
                                        /          \
                              Resolved Tickets   Support Docs

⸻

5. Frontend / Backend Separation

Streamlit acts as the frontend application.

Streamlit should not directly access SQLite, ChromaDB, or backend services.

Instead, all communication with the backend happens through the FastAPI REST API.

Streamlit
    │
    │ HTTP REST API
    ▼
FastAPI
    │
    ├── Application Services
    ├── SQLite
    └── RAG Pipeline

For example:

Streamlit
    │
    │ POST /api/v1/tickets
    ▼
FastAPI
    │
    ▼
TicketService
    │
    ▼
SQLite

This provides a clean separation between the frontend and backend.

It also allows Streamlit to be replaced by another frontend in the future without changing the core backend.

⸻

6. Backend Architecture

We will use a layered/modular architecture rather than traditional MVC.

API Layer
    ↓
Service Layer
    ↓
Repository / RAG Layer
    ↓
Infrastructure

API Layer

Implemented using FastAPI routes.

Responsibilities:

* HTTP request handling
* Request validation
* Response serialization
* Calling application services

Service Layer

Contains application and business logic.

Examples:

* TicketService
* ResolutionService
* RetrievalService

Repository Layer

Responsible for persistent application data.

Examples:

* TicketRepository
* FeedbackRepository

RAG Layer

Responsible for AI and retrieval functionality.

Examples:

* Document ingestion
* Chunking
* Embeddings
* Retrieval
* Prompt construction
* LLM interaction
* Response parsing

⸻

7. Proposed Repository Structure

rag-support-ticket-resolver/
│
├── frontend/
│   └── streamlit_app/
│       ├── app.py
│       ├── pages/
│       └── api_client.py
│
├── backend/
│   └── app/
│       ├── api/
│       │   └── tickets.py
│       │
│       ├── schemas/
│       │   └── ticket.py
│       │
│       ├── models/
│       │   └── ticket.py
│       │
│       ├── repositories/
│       │   └── ticket_repository.py
│       │
│       ├── services/
│       │   ├── ticket_service.py
│       │   ├── resolution_service.py
│       │   └── retrieval_service.py
│       │
│       ├── rag/
│       │   ├── ingestion.py
│       │   ├── chunking.py
│       │   ├── embeddings.py
│       │   ├── vector_store.py
│       │   ├── retriever.py
│       │   ├── prompts.py
│       │   └── resolution_chain.py
│       │
│       ├── db/
│       │   └── database.py
│       │
│       └── main.py
│
├── data/
│   ├── resolved_tickets/
│   ├── support_docs/
│   └── evaluation/
│
├── scripts/
│   └── ingest_knowledge_base.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── docs/
│
├── .env.example
├── .gitignore
├── requirements.txt
├── docker-compose.yml
└── README.md

⸻

8. Data Storage Responsibilities

SQLite and ChromaDB serve different purposes in the application.

SQLite

SQLite stores application state.

Examples:

* Tickets
* AI-generated resolutions
* Ticket status
* Priority
* User feedback
* Processing status
* Created and updated timestamps

Example ticket model:

Ticket
────────────────────
id
title
description
priority
status
created_at
updated_at

ChromaDB

ChromaDB stores semantic knowledge used for retrieval.

Examples:

* Resolved historical tickets
* Support documentation
* Troubleshooting guides
* Runbooks
* FAQs

Documents should contain metadata that can later be used for filtering, attribution, and conflict detection.

Example:

{
  "source_type": "resolved_ticket",
  "ticket_id": "INC-1234",
  "product": "VPN",
  "created_at": "2026-08-14"
}

Metadata will be especially useful for source attribution and staleness/conflict detection.

⸻

9. Initial REST APIs

Health Check

GET /health

Used by the frontend and deployment environment to verify that the backend is running.

Create Ticket

POST /api/v1/tickets

Example request:

{
  "title": "VPN connection fails",
  "description": "User receives authentication error after password reset.",
  "priority": "medium"
}

Example response:

{
  "id": 101,
  "title": "VPN connection fails",
  "status": "open",
  "priority": "medium"
}

List Tickets

GET /api/v1/tickets

Get Ticket

GET /api/v1/tickets/{ticket_id}

Resolve Ticket

POST /api/v1/tickets/{ticket_id}/resolve

Runs the RAG pipeline and generates a suggested resolution.

Find Similar Tickets

GET /api/v1/tickets/{ticket_id}/similar

Returns semantically similar historical tickets.

Submit Feedback

POST /api/v1/tickets/{ticket_id}/feedback

Allows users to provide feedback on the AI-generated suggestion.

⸻

10. RAG Pipeline

The initial AI workflow should remain intentionally simple.

We do not need an agent for the core implementation.

New Support Ticket
        │
        ▼
Create Embedding
        │
        ▼
Search ChromaDB
        │
        ├───────────────┐
        ▼               ▼
Similar Tickets    Support Documents
        │               │
        └───────┬───────┘
                ▼
          Top-K Context
                │
                ▼
         Prompt Template
                │
                ▼
               LLM
                │
                ▼
      Suggested Resolution
                │
                ▼
       Sources / Citations

LangChain will primarily be used inside the RAG layer.

Potential LangChain components include:

Document Loaders
      ↓
Text Splitters
      ↓
Embeddings
      ↓
Chroma Vector Store
      ↓
Retriever
      ↓
Prompt Template
      ↓
LLM
      ↓
Output Parser

FastAPI, SQLite, SQLAlchemy, and the rest of the backend remain normal Python application code.

⸻

Implementation Plan

The project will be developed incrementally using vertical slices.

⸻

Milestone 1 — Application Foundation

Goal

Streamlit
    ↓
FastAPI
    ↓
SQLite

Deliverables

* GitHub repository
* Python project setup
* FastAPI application
* Uvicorn development server
* /health endpoint
* SQLite integration
* Ticket model
* Create Ticket API
* List Tickets API
* Basic Streamlit UI
* Streamlit → FastAPI communication

Success Criteria

A user creates a ticket from Streamlit, FastAPI receives it, and the ticket is persisted in SQLite.

⸻

Milestone 2 — Knowledge Base

Build the data required for RAG.

Resolved Tickets ─┐
                  ├── Chunking
Support Docs ─────┘
                       ↓
                   Embeddings
                       ↓
                    ChromaDB

Deliverables

* Synthetic resolved-ticket dataset
* Synthetic support documentation
* Document loader
* Chunking strategy
* Embedding generation
* ChromaDB persistence
* Metadata strategy
* Retrieval testing

Success Criteria

Given a support issue, the system can retrieve relevant historical tickets and documentation.

⸻

Milestone 3 — RAG Resolution

New Ticket
    ↓
Retriever
    ↓
Relevant Context
    ↓
Prompt
    ↓
LLM
    ↓
Grounded Resolution

Deliverables

* LangChain retrieval pipeline
* Prompt template
* LLM integration
* Resolution generation
* Source attribution
* Similar-ticket results
* Resolve Ticket API
* Resolution UI

Success Criteria

A user can create a ticket and request an AI-generated resolution grounded in retrieved support information.

⸻

Milestone 4 — Conflict and Staleness Detection

The system should identify situations where retrieved information may conflict.

For example:

Older Documentation
"Reset the local VPN certificate."
             VS
Recent Resolved Ticket
"Certificate reset procedure was deprecated.
Use SSO device re-enrollment."

Instead of silently choosing one answer, the UI should warn the support engineer:

⚠ Potential documentation conflict detected.
Newer resolved tickets indicate that this
procedure may have changed.

⸻

Milestone 5 — Evaluation

We should measure whether the RAG system is actually working.

Create an evaluation dataset containing examples such as:

ticket
expected_category
expected_sources
expected_resolution

Potential evaluation metrics include:

* Retrieval Hit Rate
* Recall@K
* Source relevance
* Citation correctness
* Groundedness
* Resolution quality
* Hallucination / unsupported-answer rate

We can experiment with:

* Chunk size
* Chunk overlap
* Top-K
* Embedding models
* Prompt design
* Retrieval strategies

and compare their impact.

⸻

Initial User Stories

Epic 1 — Project Foundation

US-001 — Initialize Repository

As a developer, I want a shared GitHub repository so the team can collaborate on the project.

US-002 — Backend Bootstrap

As a developer, I want a FastAPI backend so clients can interact with the application through REST APIs.

US-003 — Frontend Bootstrap

As a user, I want a Streamlit application so I can interact with the Support Ticket Resolver.

Dependencies:

US-001
 ├── US-002
 └── US-003

⸻

Epic 2 — Ticket Management

US-004 — Ticket Persistence

As a user, I want tickets persisted so that they remain available after creation.

US-005 — Create Ticket API

As a user, I want to create a support ticket.

Dependency: US-004

US-006 — Create Ticket UI

As a user, I want to create tickets through Streamlit.

Dependency: US-005

US-007 — Ticket List

As a user, I want to view existing tickets.

⸻

Epic 3 — Knowledge Base

US-008 — Resolved Ticket Dataset

Create a controlled synthetic dataset of historical resolved tickets.

US-009 — Support Documentation Dataset

Create support documentation, troubleshooting guides, and runbooks.

US-010 — Knowledge Base Ingestion

Chunk, embed, and store knowledge-base documents in ChromaDB.

Dependencies:

US-008 ─┐
        ├── US-010
US-009 ─┘

⸻

Epic 4 — RAG Resolution

US-011 — Similar Ticket Retrieval

Retrieve historical tickets that are semantically similar to a new support ticket.

Dependency: US-010

US-012 — Grounded Resolution Generation

Generate a suggested resolution using retrieved context.

Dependency: US-011

US-013 — Resolve Ticket API

Expose RAG resolution through:

POST /api/v1/tickets/{ticket_id}/resolve

US-014 — Resolution UI

Display:

* Suggested resolution
* Similar tickets
* Supporting documentation
* Sources/citations

⸻

Epic 5 — Reliability

US-015 — Conflict Detection

Detect potentially conflicting retrieved information.

US-016 — Staleness Detection

Warn when older documentation may have been superseded by newer information.

US-017 — Insufficient Evidence Handling

Avoid generating confident answers when retrieval does not provide enough supporting evidence.

⸻

Epic 6 — Evaluation

US-018 — Evaluation Dataset

Create known ticket/resolution test cases.

US-019 — Retrieval Evaluation

Measure retrieval performance.

US-020 — Response Evaluation

Evaluate groundedness, citations, and resolution quality.

⸻

Dependency Overview

                    Repository
                        │
               ┌────────┴────────┐
               ▼                 ▼
            FastAPI          Streamlit
               │
               ▼
             SQLite
               │
               ▼
           Ticket APIs
               │
               ▼
          Working Ticket App
               │
        ┌──────┴───────┐
        ▼              ▼
Resolved Tickets   Support Docs
        └──────┬───────┘
               ▼
           Chunking
               ▼
          Embeddings
               ▼
           ChromaDB
               ▼
           Retrieval
               ▼
          LangChain RAG
               ▼
       Suggested Resolution
               ▼
        Sources/Citations
               ▼
     Conflict/Staleness Check
               ▼
            Evaluation

⸻

Suggested Team Workstreams

For a four-person team, initial ownership can be divided into four workstreams.

Workstream 1 — Platform / Backend

Responsibilities:

* Repository setup
* FastAPI
* SQLite
* SQLAlchemy
* Ticket APIs
* Service/repository architecture

Workstream 2 — Frontend

Responsibilities:

* Streamlit application
* Ticket creation
* Ticket listing
* Ticket details
* Resolution display
* Backend API client

Workstream 3 — RAG / Knowledge Base

Responsibilities:

* Synthetic data
* Document ingestion
* Chunking
* Embeddings
* ChromaDB
* Retrieval
* Similar-ticket search

Workstream 4 — AI / Evaluation

Responsibilities:

* Prompt engineering
* LangChain resolution pipeline
* Source attribution
* Conflict/staleness detection
* Evaluation dataset
* Evaluation metrics

The team should agree on API contracts and data models early so these workstreams can proceed in parallel.

⸻

First Development Target

Before implementing RAG, LangChain, embeddings, or LLM calls, the first target should be:

GitHub Repository
       ↓
Python Project
       ↓
FastAPI
       ↓
GET /health
       ↓
SQLite
       ↓
POST /tickets
       ↓
GET /tickets
       ↓
Streamlit
       ↓
Create Ticket
       ↓
Ticket persisted in SQLite

Once this vertical slice works end-to-end, the team can begin integrating the RAG pipeline.

⸻

Definition of Initial Success

A successful demo should allow someone to:

1. Open the Streamlit application.
2. Create a support ticket.
3. View the newly created ticket.
4. Request an AI-suggested resolution.
5. Retrieve relevant historical tickets.
6. Retrieve relevant support documentation.
7. See a grounded suggested resolution.
8. See the sources supporting that resolution.
9. Receive a warning when supporting information appears conflicting or stale.
10. Provide feedback on the suggested resolution.

This represents the core capstone.

Agentic functionality is not required to demonstrate the value of the project.

⸻

Future / Stretch Architecture

Only after the core project is complete should we consider capabilities such as:

* LangGraph workflow orchestration
* Tool calling
* MCP servers
* Specialized agents
* Automated ticket categorization
* Automated routing
* External ticketing integrations
* Human-in-the-loop approval workflows
* Jira / ServiceNow / Zendesk integration

These are potential extensions, not dependencies for the initial implementation.

⸻

Guiding Principle

Build the smallest complete system first.

Ticket
  ↓
Retrieve
  ↓
Generate
  ↓
Ground
  ↓
Explain
  ↓
Evaluate

Once this works reliably end-to-end, additional intelligence and orchestration can be added without changing the fundamental architecture.