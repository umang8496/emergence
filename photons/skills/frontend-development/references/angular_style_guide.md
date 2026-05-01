# Angular & Component Organization

## Standard Component Structure

All components should follow this structure:

```text
src/app/component/my-feature/
├── my-feature.component.ts       # Component logic & inputs/outputs
├── my-feature.component.html     # Template
├── my-feature.component.scss     # Component styles
└── my-feature.component.spec.ts  # Unit tests
```

## Component Template

```typescript
import { Component, Input, Output, EventEmitter, OnInit } from '@angular/core';

@Component({
  selector: 'app-my-feature',
  templateUrl: './my-feature.component.html',
  styleUrls: ['./my-feature.component.scss']
})
export class MyFeatureComponent implements OnInit {
  @Input() config: any;
  @Output() onAction = new EventEmitter<any>();

  constructor(/* inject services */) {}

  ngOnInit(): void {
    // Initialization logic
  }

  handleAction(data: any): void {
    this.onAction.emit(data);
  }
}
```

## Key Conventions

- **Selectors**: Use `app-` prefix (e.g., `app-my-feature`)
- **Inputs/Outputs**: Use `@Input()` and `@Output()` for component communication
- **Services**: Inject via constructor for dependency injection
- **Lifecycle**: Implement `OnInit`, `OnDestroy` as needed
- **Change Detection**: Use `ChangeDetectionStrategy.OnPush` for performance when possible
- **Module Imports**: Ensure component is declared in appropriate module

## File Organization

| Directory          | Purpose                                  |
|--------------------|------------------------------------------|
| src/app/component/ | Reusable & page components               |
| src/app/services/  | Services for API calls, state, utilities |
| src/app/model/     | TypeScript interfaces & types            |
| src/app/pipe/      | Custom Angular pipes                     |
| src/app/directive/ | Custom directives                        |
| src/app/helper/    | Utility functions                        |
| src/assets/        | Global styles, images, static files      |

## Naming Conventions

- **Files**: kebab-case (e.g., `my-feature.component.ts`)
- **Classes**: PascalCase (e.g., `MyFeatureComponent`)
- **Variables**: camelCase (e.g., `isVisible`, `handleClick`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_ITEMS`)
- **Selectors**: app-kebab-case (e.g., `app-my-feature`)
