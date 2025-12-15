---
trigger: always_on
---

* Make sure all the code is styled with PEP 8 style guide Ruff compatible
* Make sure all the code is properly commented

## Key Conventions
- **Language**: All code, comments, variables, and documentation MUST be in English.
- **Path Handling**: Use `pathlib` for cross-platform compatibility.
- **Logging**: Use descriptive print statements or a logger (English messages).

## Coding Standards
- **Language**: Python 3.12+
- **Style**: Follow PEP 8.
- **Type Hinting**: Use strict type hinting for all function signatures.

## SOLID Principles
- **Single Responsibility Principle (SRP)**: Each class or module should have one, and only one, reason to change.
- **Open/Closed Principle (OCP)**: Software entities should be open for extension, but closed for modification.
- **Liskov Substitution Principle (LSP)**: Subtypes must be substitutable for their base types.
- **Interface Segregation Principle (ISP)**: Clients should not be forced to depend on interfaces they do not use.
- **Dependency Inversion Principle (DIP)**: High-level modules should not depend on low-level modules. Both should depend on abstractions.
- **Dependency Injection**: Pass dependencies (like `FileManager`) explicitly to functions/classes rather than creating them inside.