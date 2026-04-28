# SCSS Patterns & Styling Conventions

## Global Styling

The project uses a combination of:

- **Material Design**: Via Angular Material (pre-built components and themes)
- **Custom SCSS**: Project-specific styles in `src/assets/` and component-scoped SCSS

## SCSS Organization

```text
src/assets/
├── _nav-links.scss        # Navigation styling
├── _variables.scss        # Global color, spacing, typography variables
├── _utilities.scss        # Reusable utility classes
└── styles.scss            # Global imports

src/app/component/*/
└── *.component.scss       # Component-scoped styles
```

## Using Variables

Always use SCSS variables for consistency:

```scss
// Colors
$primary-color: #007bff;
$secondary-color: #6c757d;
$success-color: #28a745;
$error-color: #dc3545;

// Spacing (8px grid)
$spacing-xs: 4px;
$spacing-sm: 8px;
$spacing-md: 16px;
$spacing-lg: 24px;
$spacing-xl: 32px;

// Typography
$font-size-base: 14px;
$font-size-large: 18px;
$font-family-base: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;

// Breakpoints
$breakpoint-mobile: 320px;
$breakpoint-tablet: 768px;
$breakpoint-desktop: 1024px;
```

## Responsive Design Mixins

```scss
// Mobile-first approach
@mixin respond-tablet {
  @media (min-width: 768px) {
    @content;
  }
}

@mixin respond-desktop {
  @media (min-width: 1024px) {
    @content;
  }
}

// Usage
.component {
  font-size: 14px;
  
  @include respond-tablet {
    font-size: 16px;
  }
  
  @include respond-desktop {
    font-size: 18px;
  }
}
```

## BEM Naming (when not using component scope)

```scss
// Block__Element--Modifier
.card {
  padding: $spacing-md;
  
  &__header {
    font-weight: bold;
  }
  
  &__title {
    font-size: $font-size-large;
  }
  
  &__content {
    margin-top: $spacing-md;
  }
  
  &--featured {
    border: 2px solid $primary-color;
  }
}
```

## Component-Scoped SCSS

Component styles are automatically scoped and don't need additional selectors:

```scss
// my-feature.component.scss
// Styles here only apply to this component

.container {
  padding: $spacing-md;
  background: $secondary-color;
}

.title {
  font-size: $font-size-large;
  font-weight: bold;
  color: $primary-color;
}

// Responsive
@include respond-tablet {
  .container {
    padding: $spacing-lg;
  }
}
```

## Common Patterns

### Flexbox Layout

```scss
.flex-container {
  display: flex;
  gap: $spacing-md;
  align-items: center;
  justify-content: space-between;
  
  @include respond-mobile {
    flex-direction: column;
  }
}
```

### Grid Layout

```scss
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: $spacing-md;
}
```

### Accessibility

```scss
// Visible focus states for keyboard navigation
button:focus-visible {
  outline: 2px solid $primary-color;
  outline-offset: 2px;
}

// Hide visually but keep for screen readers
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
}
```

## Avoid

- Inline styles in templates
- Deep nesting (more than 3 levels)
- Colors without variables
- Magic numbers for spacing/sizing
- `!important` unless absolutely necessary
- Targeting by tag name globally (use classes)
