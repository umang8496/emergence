---
name: code-review-qa
description: 'Expert code review for frontend development. Use when: reviewing TypeScript/Angular code for null/undefined errors and type safety, checking HTML/CSS for accessibility and consistency, validating error handling and cleanup logic, detecting memory leaks and performance bottlenecks, finding dead code and code duplication. Acts as experienced software engineer conducting quality assurance before code submission.'
---

# Code Review & Quality Assurance

## When to Use

- Reviewing TypeScript/Angular code for runtime errors before submission
- Validating HTML/CSS against design philosophy and accessibility standards
- Checking for memory leaks, unsubscribed observables, and resource cleanup
- Identifying null/undefined object errors and type safety issues
- Detecting dead code, duplication, and maintainability issues
- Validating error handling coverage and edge cases
- Performance review and optimization opportunities
- Post-implementation quality gate before merging code

## Core Principles

1. **Catch errors before runtime** - Identify null pointer, undefined, and type errors proactively
2. **Respect developer expertise** - Provide insights and suggestions, not directives
3. **Comprehensive coverage** - Review across TypeScript, HTML, CSS, and architecture
4. **Best practices enforcement** - Align with project conventions and industry standards
5. **Performance awareness** - Flag potential performance issues early

## Review Workflow

### Step 1: Scope Assessment

1. **Identify what's being reviewed**
   - New component or feature modification?
   - Single file or multiple files?
   - TypeScript logic, template markup, styles, or combination?

2. **Understand context**
   - What is this code supposed to do?
   - What are the entry points and data flows?
   - Any external dependencies or API calls?

### Step 2: TypeScript/Logic Review

#### Null Safety & Type Safety

- [ ] **Null/Undefined Checks**
  - Are all object property accesses guarded? (e.g., `obj?.property` or null check before access)
  - Do all API responses handle null/undefined returns?
  - Are optional properties (`property?: type`) handled in logic?
  - Do array accesses check length before indexing?
  - Observable subscriptions properly handled (unsubscribe or takeUntil)?

- [ ] **Type Safety**
  - All variables have explicit or inferable types?
  - Function parameters and return types are typed?
  - Cast operations justified and safe?
  - Generic types properly constrained?
  - Any `any` types that should be specific?

#### Code Example - Issues to Catch

```typescript
// UNSAFE: No null check
const userName = user.profile.name;

// SAFE: Guard against undefined
const userName = user?.profile?.name ?? 'Unknown';

// UNSAFE: Array access without bounds check
const firstItem = items[0].id;

// SAFE: Check length first
const firstItem = items.length > 0 ? items[0].id : null;

// UNSAFE: Unsubscribed observable
this.service.getData().subscribe(data => this.data = data);

// SAFE: Proper cleanup
this.service.getData()
  .pipe(takeUntil(this.destroy$))
  .subscribe(data => this.data = data);
```

#### Error Handling

- [ ] **Try-Catch Coverage**
  - All async operations have error handlers?
  - Promise rejections are caught?
  - Error objects properly typed and handled?
  - Meaningful error messages logged?

- [ ] **API Calls**
  - Request failures handled?
  - Timeout scenarios covered?
  - Rate limiting or retry logic needed?
  - Error states reflected in UI?

#### Memory Leaks & Cleanup

- [ ] **Subscription Management**
  - All observables have unsubscribe patterns (takeUntil, unsubscribe in ngOnDestroy)?
  - Event listeners removed (removeEventListener)?
  - Timers cleared (clearTimeout, clearInterval)?
  - Component destruction logic present (ngOnDestroy)?

- [ ] **Resource Cleanup**
  - File handles or streams properly closed?
  - Third-party library cleanups called?
  - DOM references cleared to avoid memory retention?

#### Performance Issues

- [ ] **Change Detection**
  - Unnecessary change detection cycles?
  - Consider OnPush strategy where possible?
  - Heavy computations in templates (use pipes or component logic)?
  - Tracking functions for *ngFor lists?

- [ ] **Data & Rendering**
  - Large arrays rendered without pagination/virtualization?
  - Unnecessary API calls or duplicate requests?
  - Images or assets optimized?
  - DOM queries in loops?

#### Dead Code & Duplication

- [ ] **Unused Code**
  - Imported but unused modules/services?
  - Unused variables or function parameters?
  - Dead branches or unreachable code?
  - Old commented-out code that should be removed?

- [ ] **Code Duplication**
  - Similar logic repeated in multiple places?
  - Extract to shared service or utility function?
  - Could components be composed differently to reduce duplication?

### Step 3: Template (HTML) Review

#### Accessibility

- [ ] **Semantic HTML**
  - Using appropriate HTML elements (button for buttons, not div)?
  - Form fields have associated labels?
  - Headings are in logical order (h1, h2, h3)?
  - Lists use proper list elements (ul/li)?

- [ ] **ARIA & Labels**
  - aria-label or aria-labelledby on interactive elements?
  - aria-required on required form fields?
  - aria-invalid on validation errors?
  - role attributes only when semantic element unavailable?

- [ ] **Keyboard Navigation**
  - All interactive elements focusable (buttons, links, inputs)?
  - Tab order logical?
  - Focus traps avoided?
  - Keyboard shortcuts not conflicting with browser defaults?

- [ ] **Screen Reader Support**
  - Images have alt text (or alt="" if decorative)?
  - Form errors announced?
  - Live regions for dynamic content (aria-live)?
  - Skip links for main content?

#### Data Binding & Safety

- [ ] **Template Expressions**
  - Null/undefined safe operator used (`?.`) where appropriate?
  - Async operations use async pipe or unsubscribe pattern?
  - Event handlers properly bound?
  - No side effects in template expressions?

- [ ] **Form Binding**
  - Two-way binding appropriate for the use case?
  - Form validation state reflected in UI?
  - Error messages shown contextually?
  - Required fields indicated?

#### Performance

- [ ] **Change Detection**
  - OnPush change detection compatible?
  - TrackBy functions on *ngFor lists?
  - Complex computations in component class, not template?

- [ ] **Rendering Efficiency**
  - Unnecessary DOM elements?
  - Hidden elements still in DOM (consider *ngIf vs hidden)?

### Step 4: Styles (CSS/SCSS) Review

#### Consistency & Standards

- [ ] **Design System Alignment**
  - Using project SCSS variables for colors/spacing?
  - Consistent naming conventions (BEM or component-scoped)?
  - No magic numbers (use variables)?
  - Proper use of Flexbox/Grid?

- [ ] **Responsive Design**
  - Mobile-first approach applied?
  - All breakpoints tested?
  - Touch targets >= 48px?
  - No horizontal scrolling on mobile?

#### Performance & Maintenance

- [ ] **CSS Quality**
  - Avoid deep nesting (max 3 levels)?
  - Avoid `!important` unless absolutely necessary?
  - No inline styles in templates?
  - No unused CSS rules?

- [ ] **SCSS Best Practices**
  - Mixins used for repeated patterns?
  - Variables used for colors/sizing/spacing?
  - Component-scoped styles (not global)?

### Step 5: Architecture & Patterns

- [ ] **Component Design**
  - Single responsibility principle followed?
  - Proper separation of concerns?
  - Services handle logic, components handle presentation?
  - Dependency injection used correctly?

- [ ] **Reusability**
  - Components too specific or can be made more generic?
  - Props/inputs well-designed for reuse?
  - Hard-coded strings/values extracted to inputs?

- [ ] **Module Organization**
  - Components declared in appropriate modules?
  - Lazy loading applied where beneficial?
  - Unnecessary circular dependencies?

### Step 6: Testing & Documentation

- [ ] **Test Coverage**
  - Critical paths have tests?
  - Edge cases covered (null, empty arrays, errors)?
  - Mocks used appropriately for services?

- [ ] **Documentation**
  - Complex logic explained in comments?
  - Public APIs documented (props, methods)?
  - README updated for new components?

## Common Anti-patterns to Catch

### TypeScript/JavaScript

```typescript
// No null check - runtime error risk
const email = user.contact.email.toLowerCase();

// Floating promise - not awaited
this.dataService.save(data);

// Memory leak - observable not unsubscribed
this.route.params.subscribe(params => this.load(params.id));

// Performance - function called on every change detection
{{ getFormattedDate(item.date) }}

// Type unsafe
const data: any = response;
```

### HTML/Template

```html
<!-- Missing alt text -->
<img src="logo.png">

<!-- No label association -->
<input type="text" placeholder="Email">

<!-- Keyboard inaccessible -->
<div (click)="handleClick()">Click me</div>

<!-- Unsafe binding -->
<div>{{ user.name }}</div>  <!-- What if user is null? -->

<!-- Performance - no trackBy -->
<div *ngFor="let item of items">{{ item.name }}</div>
```

### CSS/SCSS

```scss
// Magic number instead of variable
padding: 16px;

// Deep nesting
.container {
  .header {
    .title {
      .text {
        color: red;
      }
    }
  }
}

// Inline styles in template
<div style="color: blue; padding: 10px;">Text</div>

// No responsive design
width: 1200px;  // Fixed width, won't work on mobile
```

## Review Checklist

Quick reference for code review:

- [ ] **Safety First**
  - [ ] No null/undefined access without guards
  - [ ] All observables properly unsubscribed
  - [ ] Error handling for async operations
  - [ ] Type safety enforced

- [ ] **Performance**
  - [ ] No memory leaks
  - [ ] Change detection optimized
  - [ ] No unnecessary renders or API calls
  - [ ] Data structures appropriate for use

- [ ] **Code Quality**
  - [ ] No dead code or duplication
  - [ ] Clear variable/function names
  - [ ] Follows project conventions
  - [ ] Properly tested

- [ ] **Accessibility**
  - [ ] Semantic HTML
  - [ ] Keyboard navigable
  - [ ] Screen reader friendly
  - [ ] ARIA labels where needed

- [ ] **Design Alignment**
  - [ ] Uses design system tokens
  - [ ] Responsive at all breakpoints
  - [ ] Consistent with existing UI
  - [ ] Visual hierarchy clear

## Automation Tools & Integration

### Linting & Type Checking

Run these before review to catch automated issues:

```bash
# TypeScript type checking
npx tsc --noEmit

# ESLint for code quality
npm run lint

# SCSS linting
npm run lint:styles

# Angular specific checks
ng lint
```

### IDE Integration

Configure IDE/Editor to catch issues during development:

- **VS Code Settings**
  - Enable strict null checks in tsconfig.json
  - Use ESLint extension for real-time linting
  - Enable Angular Language Service
  - Use Prettier for auto-formatting

### Checklist Tools

Suggest to developers:

- Use TypeScript strict mode (recommended for new code)
- Enable ESLint rules for performance (no-unneeded-ternary, etc.)
- Use `ng-strict-template-check` for template safety
- Enable a11y linting (eslint-plugin-jsx-a11y equivalents)

## Review Output

When reviewing code, provide:

1. **Critical Issues** (must fix before merge)
   - Runtime errors (null access, type errors)
   - Memory leaks
   - Security vulnerabilities
   - Accessibility failures

2. **Important Issues** (should fix)
   - Performance problems
   - Error handling gaps
   - Code duplication
   - Design inconsistency

3. **Suggestions** (nice to have)
   - Refactoring opportunities
   - Better naming
   - Code organization
   - Performance optimizations

4. **Positive Notes** (what's good)
   - Well-written code
   - Good test coverage
   - Clean architecture

## Resources

- [Common TypeScript Pitfalls](./references/typescript-pitfalls.md)
- [Memory Leak Patterns](./references/memory-leaks.md)
- [Performance Optimization Guide](./references/performance-guide.md)
- [Accessibility Checklist](./references/a11y-checklist.md)
- [Angular Best Practices](./references/angular-best-practices.md)
- [ESLint Rules Reference](./references/eslint-rules.md)

## Key Principle

**Improve code quality through collaborative feedback.**  
Frame reviews as learning opportunities and suggestions, not requirements.  
Highlight both issues and strengths to build a positive review culture.
