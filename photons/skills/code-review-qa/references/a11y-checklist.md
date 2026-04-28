# Accessibility Checklist

Ensure code meets WCAG accessibility standards and is inclusive for all users.

---

## Semantic HTML

- [ ] **Correct Elements Used**
  - Buttons: `<button>`, not `<div onclick="...">`
  - Links: `<a href="...">`, not `<span onclick="...">`
  - Headings: `<h1>`, `<h2>`, etc., not `<p class="heading">`
  - Lists: `<ul>`, `<ol>`, `<li>`, not `<div>` with custom styling

- [ ] **Heading Hierarchy**
  - Headings are in logical order (no skipping h1 to h3)
  - One `<h1>` per page
  - Headings accurately describe content

- [ ] **Form Structure**
  - All form inputs have associated labels: `<label for="input-id">`
  - Labels describe input purpose
  - Required fields marked with `aria-required="true"`
  - Related fields grouped with `<fieldset>` and `<legend>`

---

## Keyboard Navigation

- [ ] **All Interactive Elements Focusable**
  - Buttons, links, inputs are accessible via Tab
  - Custom interactive components have `tabindex="0"`
  - Buttons/links never have `tabindex="-1"` unless intentional

- [ ] **Focus Management**
  - Focus order is logical and intuitive
  - Focus indicator clearly visible (not removed)
  - Focus trap avoided (can Tab out of modals with Esc or close button)
  - Modal focus trapped internally when open

- [ ] **Keyboard Shortcuts**
  - Common shortcuts: Enter (submit), Esc (close), Spacebar (toggle)
  - Custom shortcuts documented
  - No conflicting with browser shortcuts (Ctrl+S, Ctrl+F, etc.)
  - Arrow keys for navigation in lists/menus

---

## Visual Design & Contrast

- [ ] **Color Contrast (WCAG AA)**
  - Normal text: 4.5:1 ratio minimum
  - Large text (18pt+): 3:1 ratio minimum
  - Check using WebAIM contrast checker

- [ ] **Color Not Only Indicator**
  - Information conveyed with color also uses text/icons
  - Error states: red color + error icon/text
  - Success states: green color + checkmark/text
  - Links: color + underline or other indicator

- [ ] **Focus Indicators**
  - Default focus outline visible (2px outline recommended)
  - High contrast focus state (not just opacity change)
  - Focus visible on keyboard navigation

- [ ] **Responsive Text**
  - Text doesn't require horizontal scrolling at 200% zoom
  - Text reflows appropriately on zoom

## Images & Media

- [ ] **Image Alt Text**
  
  ```html
  <!-- Meaningful alt text -->
  <img src="user.jpg" alt="Sarah Johnson, Product Manager">
  
  <!-- Decorative images -->
  <img src="divider.jpg" alt="">
  
  <!-- Avoid "image of", "picture of" in alt text -->
  ```

- [ ] **Icon Images**
  - SVG or icon fonts have `aria-label` or `role="img"`
  - Icon fonts include `aria-hidden="true"` if decorative

- [ ] **Media & Video**
  - Video has captions (for deaf/hard of hearing)
  - Video has audio description (for blind/low vision)
  - Auto-play avoided or controlled
  - Avoid flashing content (more than 3x per second)

---

## ARIA & Labels

- [ ] **Proper ARIA Usage**
  - Only use ARIA when semantic HTML insufficient
  - `aria-label` for icon-only buttons
  - `aria-labelledby` for titles/descriptions
  - `aria-describedby` for additional help text

```html
<!-- Icon button -->
<button aria-label="Close menu">
  <i class="icon-close"></i>
</button>

<!-- Described input -->
<input id="password" type="password" 
       aria-describedby="pwd-hint">
<span id="pwd-hint">Password must be 8+ characters</span>

<!-- Navigation landmark -->
<nav aria-label="Main navigation">
  <ul><!-- menu items --></ul>
</nav>
```

- [ ] **Live Regions**
  - Dynamic content uses `aria-live="polite"` or `aria-live="assertive"`
  - Form errors announced: `role="alert"`
  - Loading states announced

```html
<!-- Error alert -->
<div role="alert">Email is invalid</div>

<!-- Live region for updates -->
<div aria-live="polite" aria-atomic="true">
  5 results found
</div>
```

---

## Forms & Validation

- [ ] **Error Messages**
  - Associated with input: `aria-describedby`
  - Clearly describe what's wrong
  - Suggest how to fix
  - Announced to screen readers

```html
<input id="email" type="email" 
       aria-describedby="email-error">
<span id="email-error" role="alert">
  Email must be valid (example@domain.com)
</span>
```

- [ ] **Required Fields**
  - Marked with `aria-required="true"` or `required` attribute
  - Visual indicator (asterisk) with explanation
  - Not indicated by color alone

- [ ] **Form Labels**
  - Explicit labels: `<label for="id">`
  - Placeholder is not substitute for label
  - Instructions placed before input

```html
<!-- WRONG -->
<input placeholder="Enter email">

<!-- RIGHT -->
<label for="email">Email</label>
<input id="email" type="email">
```

## Dynamic Content & Updates

- [ ] **Async Content Loading**
  - Loading state communicated: `aria-busy="true"`
  - Updates announced to screen readers
  - Content doesn't surprise user (focus management)

- [ ] **Modals & Dialogs**
  - `role="dialog"` with `aria-modal="true"`
  - Focus trapped inside modal
  - Esc key closes modal
  - Close button provided

```html
<div role="dialog" aria-modal="true" aria-labelledby="dialog-title">
  <h2 id="dialog-title">Confirm Delete</h2>
  <p>Are you sure?</p>
  <button>Cancel</button>
  <button>Delete</button>
</div>
```

- [ ] **Notifications/Toasts**
  - Use `role="alert"` for important messages
  - `aria-live="assertive"` for urgent notifications
  - Clear dismiss mechanism
  - Don't auto-close with keyboard users (4 sec minimum)

---

## Lists & Navigation

- [ ] **Navigation Structure**
  - Main navigation in `<nav>` landmark
  - Skip navigation link to main content
  - Current page indicated in navigation

```html
<nav aria-label="Main navigation">
  <ul>
    <li>
      <a href="/" aria-current="page">Home</a>
    </li>
    <li>
      <a href="/about">About</a>
    </li>
  </ul>
</nav>
```

- [ ] **Lists**
  - Proper list markup: `<ul>`, `<ol>`, `<li>`
  - Nested lists properly structured
  - Description lists `<dl>`, `<dt>`, `<dd>` for key-value pairs

---

## Testing Accessibility

### Manual Testing

- [ ] Test with keyboard only (Tab, Shift+Tab, Enter, Esc, Arrow keys)
- [ ] Test with screen reader (NVDA on Windows, VoiceOver on Mac/iOS)
- [ ] Zoom to 200% and verify layout
- [ ] Check with browser zoom to 200%
- [ ] Test on mobile with accessibility features (VoiceOver, TalkBack)

### Tools

- **Wave Browser Extension**: Identifies accessibility issues
- **Axe DevTools**: Automated accessibility testing
- **NVDA/JAWS**: Free/commercial screen readers
- **Chrome Lighthouse**: Accessibility audit
- **WebAIM Contrast Checker**: Color contrast validation

### Browser Testing

- [ ] Chrome with Axe DevTools extension
- [ ] Firefox with WAVE extension
- [ ] Safari with VoiceOver (macOS)
- [ ] Mobile Safari with VoiceOver (iOS)

---

## Code Accessibility Checklist

**Navigation & Structure**  

- [ ] Proper semantic HTML used
- [ ] Heading hierarchy logical
- [ ] Navigation landmarks present
- [ ] Skip link to main content

**Keyboard Access**  

- [ ] All interactive elements keyboard accessible
- [ ] Focus order logical and visible
- [ ] No keyboard traps
- [ ] Common shortcuts work

**Visual Design**  

- [ ] Text contrast meets WCAG AA (4.5:1)
- [ ] Not dependent on color alone
- [ ] Focus indicators clearly visible
- [ ] Text readable at 200% zoom

**Images & Media**  

- [ ] Meaningful images have alt text
- [ ] Decorative images have empty alt
- [ ] Videos have captions
- [ ] No auto-playing audio

**Forms**  

- [ ] All inputs have associated labels
- [ ] Error messages clear and associated
- [ ] Required fields marked
- [ ] Form instructions provided

**Dynamic Content**  

- [ ] Loading states announced
- [ ] Updates accessible to screen readers
- [ ] Modals managed properly
- [ ] Live regions for updates

**ARIA**  

- [ ] ARIA used only when semantic HTML insufficient
- [ ] Roles, states, properties correct
- [ ] aria-labels descriptive
- [ ] No conflicting ARIA

**Testing**  

- [ ] Tested with keyboard only
- [ ] Tested with screen reader
- [ ] Tested at 200% zoom
- [ ] Lighthouse accessibility audit passing
