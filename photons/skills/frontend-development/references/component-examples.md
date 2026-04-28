# Reference Component Examples

Examine these existing components in the codebase for patterns and best practices.

## Where to Find Examples

Look in `src/app/component/` directory for real examples of:

1. **Simple reusable components**
   - Form inputs and fields
   - Buttons and action components
   - Cards and containers
   - Badges and labels

2. **Page-level components**
   - Full page layouts
   - Multi-section pages
   - Composite components combining multiple sub-components

3. **Complex interactive components**
   - Data tables
   - Dropdowns/modals
   - Navigation
   - Forms with validation

## Pattern to Follow

When creating a new component, choose a similar existing component and:

1. Copy its structure (folders, file organization)
2. Use the same import/export patterns
3. Follow the same SCSS organization
4. Apply similar component input/output patterns
5. Match the overall code style and formatting

## Questions to Ask Yourself

- Is there a component that does something similar I can reference?
- Are there shared styles or utilities I should use?
- Does this component need to be reusable or is it page-specific?
- What existing components can I compose together?
- Are there any Material Design components that could accelerate this?

## Checklist Before Implementation

- [ ] Found a similar existing component to reference
- [ ] Matched the file structure and organization
- [ ] Used the same module imports and patterns
- [ ] Applied component-scoped SCSS like existing components
- [ ] Followed naming conventions from the codebase
- [ ] Used the same service injection patterns
- [ ] Validated component interfaces/props match patterns used elsewhere

## Reusability Patterns

### Single Responsibility

Each component should do one thing well. Break complex UIs into smaller, focused components.

### Props-Based Configuration

Use `@Input()` properties to make components flexible:

```typescript
@Input() variant: 'primary' | 'secondary' = 'primary';
@Input() disabled: boolean = false;
@Input() size: 'small' | 'medium' | 'large' = 'medium';
```

### Content Projection

Use `<ng-content>` to allow flexible content:

```html
<!-- card.component.html -->
<div class="card">
  <ng-content></ng-content>
</div>
```

### Composition Over Inheritance

Build complex components from simpler ones rather than extending base classes.

### Service Injection

Share logic across components via services:

```typescript
constructor(private apiService: ApiService) {}
```
