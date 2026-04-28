---
name: frontend-development
description: 'Frontend development expert for Angular and CSS work. Use when: implementing UI designs from Figma/design documents, building new components, modifying existing features, ensuring design consistency, validating responsive layouts, or improving component reusability. Asks clarifying questions before code changes and aligns implementations with the app design philosophy.'
---

# Frontend Development

## When to Use

- Translating UI designs (Figma, screenshots, mockups) into HTML/CSS/TypeScript code
- Building new Angular components aligned with existing architecture
- Modifying features with UI/styling changes
- Validating responsive design and cross-browser compatibility
- Ensuring component reusability and design consistency
- Debugging layout or styling issues
- Reviewing code changes against design philosophy

## Design Philosophy Reference

The Slovakia Onboarding web app uses:
- **Framework**: Angular with TypeScript
- **Styling**: Combination of Angular Material and custom SCSS
- **Architecture**: Component-based with reusable services
- **Design Consistency**: SCSS variables, shared utility classes, consistent theming
- **Responsive Design**: Mobile-first approach with Material Design breakpoints

## Procedure: Design to Implementation

### Phase 1: Discovery & Clarification

Before making any code changes, follow these steps:

1. **Review the requirement/design artifact**
   - Examine any attached Figma designs, screenshots, or written requirements
   - Identify target layout, visual hierarchy, colors, spacing, typography

2. **Ask clarifying questions** - DO NOT ASSUME. Common clarifications:
   - Is this a new component or modifying an existing one?
   - What is the component's scope? (page-level, reusable widget, form field, etc.)
   - Are there existing similar components I should align with?
   - What breakpoints/devices should this support? (mobile, tablet, desktop)
   - Any accessibility requirements? (ARIA labels, keyboard navigation, screen readers)
   - What data/inputs does this component need?
   - Are there any animations or interactions?
   - Which design system tokens (colors, spacing, fonts) should apply?

### Phase 2: Design Consistency Check

1. **Explore existing components**
   - Search for similar components in `src/app/component/` directory
   - Check styling patterns in `src/assets/` and existing SCSS files
   - Review shared styles and CSS variables

2. **Verify alignment**
   - Does the design use standard Material Design or custom styling?
   - What spacing/sizing system is being used? (8px grid, rem units, etc.)
   - Are there existing color tokens or should new ones be created?
   - How do existing responsive patterns work?

3. **Ask for decisions** if there are conflicts:
   - Should this follow Material Design conventions or custom styling?
   - If creating new styles, where should they live? (component.scss, shared utilities, or assets/)
   - For responsive behavior, should we follow existing patterns or create a new approach?

### Phase 3: Component Reusability Assessment

1. **Identify reuse potential**
   - Can parts of this component be extracted into smaller, reusable pieces?
   - Should this component accept multiple content types or configurations?
   - Would prop-based variations reduce code duplication elsewhere?

2. **Ask about reusability constraints**
   - Is this a one-off component or likely to be used in multiple places?
   - Should there be multiple variants? (sizes, states, themes)
   - Any future features that might affect this component's structure?

### Phase 4: Implementation

1. **Follow existing patterns**
   - Use existing component structure as reference (*.component.ts, *.component.html, *.component.scss)
   - Leverage services and dependency injection as the codebase does
   - Apply routing, lazy loading patterns already in place

2. **Responsive design validation**
   - Check at standard breakpoints (mobile 320px, tablet 768px, desktop 1024px+)
   - Use existing Material Design breakpoints if available
   - Verify touch targets are >= 48px for mobile accessibility
   - Test text readability at all breakpoints

3. **Code organization**
   - Component files in appropriate subdirectory under `src/app/component/`
   - SCSS follows component co-location pattern
   - Leverage `src/assets/_nav-links.scss` and shared utilities
   - Use SCSS variables for theming and consistency

4. **Quality checks before submission**
   - ✓ Design matches the provided mockup
   - ✓ Code follows existing Angular/TypeScript conventions
   - ✓ Responsive behavior validated at key breakpoints
   - ✓ Accessibility standards met (keyboard navigation, labels, contrast)
   - ✓ Similar components not duplicated - reuse where appropriate
   - ✓ SCSS uses project variables and patterns
   - ✓ Component is self-contained and testable

### Phase 5: Ask Before Major Decisions

Stop and ask the user before making these choices:

- **Architecture questions**: Should this be a shared component, page component, or internal widget?
- **Styling approach**: CSS-in-JS, BEM naming, utility-first, or follow existing SCSS patterns?
- **State management**: Component state vs service vs app store?
- **Performance trade-offs**: Any specific performance requirements?
- **Browser support**: Which browsers/versions must be supported?
- **Backwards compatibility**: Will this change break existing usage of related components?

## Common Scenarios

### Scenario 1: Implementing a Figma Design

```text
1. Review Figma screenshot and identify all elements
2. Ask: "Should I create a new component or use existing ones?"
3. Check for similar components in the codebase
4. Ask: "Any specific Material Design patterns or custom styling to follow?"
5. Implement following the Design Consistency Check steps
6. Validate responsive behavior
```

### Scenario 2: Feature Modification

```text
1. Locate the existing component
2. Ask: "What specific UI change is needed? How does it affect responsive behavior?"
3. Check if modification breaks existing usage
4. Ask: "Should this be a configuration prop or a new variant?"
5. Update styles and template aligned with existing patterns
6. Validate no regression in other views using this component
```

### Scenario 3: New Reusable Component

```text
1. Ask: "Where will this be used? What configurations are needed?"
2. Review similar components for patterns
3. Design props/inputs to maximize reusability
4. Ask: "Should there be multiple sizes/variants?"
5. Implement with flexibility in mind
6. Document usage with prop descriptions
```

## Resources

- [Angular Style Guide](./references/angular-style-guide.md) - Component organization
- [SCSS Patterns](./references/scss-patterns.md) - Styling conventions used in the project
- [Responsive Design Checklist](./references/responsive-checklist.md) - Validation points
- [Component Examples](./references/component-examples.md) - Reference implementations

## Key Principle

**Ask, don't assume.**  
When requirements are unclear, design decisions unclear, or trade-offs exist, stop and ask the user to decide.  
This prevents rework and ensures solutions match the user's intent and the app's architecture.
