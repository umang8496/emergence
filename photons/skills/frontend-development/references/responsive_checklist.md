# Responsive Design Validation Checklist

Use this checklist to validate responsive behavior before submitting code changes.

## Device/Breakpoint Testing

- [ ] **Mobile** (320px - 480px)
  - [ ] Layout is single-column or properly stacked
  - [ ] Touch targets are >= 48px x 48px
  - [ ] Text is readable without zooming
  - [ ] Horizontal scrolling not present (unless intentional)

- [ ] **Tablet** (768px - 1024px)
  - [ ] Two-column layouts work well
  - [ ] Navigation adapts appropriately
  - [ ] Spacing is balanced at this size

- [ ] **Desktop** (1024px+)
  - [ ] Multi-column layouts display correctly
  - [ ] Whitespace usage is appropriate
  - [ ] Content width is optimal (not too wide)

## Typography & Readability

- [ ] Font sizes scale appropriately at each breakpoint
- [ ] Line-height is sufficient (1.5-1.6 for body text)
- [ ] Line length is optimal (50-75 characters per line)
- [ ] All text meets contrast requirements (WCAG AA minimum)
- [ ] Font weights are clear and not too thin

## Layout & Spacing

- [ ] Margins and padding are consistent
- [ ] Gutters between columns are appropriate
- [ ] No content is cut off or hidden unintentionally
- [ ] Flex/grid layouts wrap correctly at breakpoints
- [ ] Whitespace doesn't feel cramped or excessive

## Navigation & Interaction

- [ ] Menu/navigation collapses/expands appropriately
- [ ] Touch targets meet minimum sizes (48px minimum)
- [ ] Hover states are appropriate for desktop
- [ ] Keyboard navigation works across devices
- [ ] Focus indicators are visible

## Images & Media

- [ ] Images scale responsively (max-width: 100%)
- [ ] Images don't cause layout shifts
- [ ] Aspect ratios are maintained
- [ ] Mobile images load efficiently (consider srcset)
- [ ] Background images scale properly

## Forms & Input

- [ ] Form fields are appropriately sized
- [ ] Labels are associated with inputs
- [ ] Error messages are visible and clear
- [ ] Mobile keyboards don't obscure critical content
- [ ] Input types use proper HTML5 attributes (number, email, tel, etc.)

## Orientation Changes

- [ ] Layout adapts when switching portrait ↔ landscape
- [ ] No content permanently hidden in either orientation
- [ ] Text reflows properly

## Browser Compatibility

- [ ] Chrome/Chromium (latest)
- [ ] Firefox (latest)
- [ ] Safari (iOS & macOS)
- [ ] Edge (latest)

## Performance

- [ ] CSS doesn't cause layout thrashing
- [ ] Smooth scrolling on mobile
- [ ] Animations are smooth (60fps target)
- [ ] No unnecessary media queries

## Accessibility

- [ ] Color alone is not used to convey information
- [ ] Sufficient contrast for text (WCAG AA minimum: 4.5:1 for normal text)
- [ ] Focus order is logical
- [ ] Labels are present for form fields
- [ ] Images have alt text where appropriate

## Tools for Testing

- **Browser DevTools**: Test responsive modes
- **Chrome DevTools**: Device emulation, lighthouse accessibility audit
- **Firefox DevTools**: Responsive mode with test devices
- **Real devices**: Test on actual phones, tablets when possible
- **Wave Browser Extension**: Accessibility testing
- **Axe DevTools**: Automated accessibility checks

## Common Responsive Issues to Check

- [ ] Text overflow in narrow containers
- [ ] Images breaking layouts (use max-width: 100%)
- [ ] Flexbox/grid not wrapping at breakpoints
- [ ] Fixed widths on responsive elements
- [ ] Forgotten media queries
- [ ] Viewport meta tag present: `<meta name="viewport" content="width=device-width, initial-scale=1">`
