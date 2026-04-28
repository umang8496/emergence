# Performance Optimization Guide

Identify and optimize performance bottlenecks during code review.

## Change Detection Performance

### Issue: Unnecessary Change Detection Cycles

```typescript
// INEFFICIENT: Default change detection on every action
@Component({
  selector: 'app-list',
  template: `...`,
})
export class ListComponent {
  items: Item[];
  // Every async action triggers change detection
}

// OPTIMIZED: OnPush change detection strategy
@Component({
  selector: 'app-list',
  template: `...`,
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class ListComponent {
  @Input() items: Item[];
  // Only updates when @Input changes or events occur
}
```

### Issue: Heavy Computations in Templates

```typescript
// INEFFICIENT: Called on every change detection cycle
<div>{{ complexCalculation(item) }}</div>

// OPTIMIZED: Computed in component class
computed$ = this.items$.pipe(
  map(items => items.map(item => ({
    ...item,
    computed: this.complexCalculation(item)
  })))
);

// Template
<div *ngFor="let item of (computed$ | async)">
  {{ item.computed }}
</div>

// OPTIMIZED: Use pure pipe
<div>{{ item | customPipe }}</div>
```

### Issue: Missing TrackBy Function

```typescript
// INEFFICIENT: Recreates all DOM elements when list changes
<div *ngFor="let item of items">
  {{ item.name }}
</div>

// OPTIMIZED: Reuses DOM elements
trackByFn(index: number, item: Item): number {
  return item.id;
}

<div *ngFor="let item of items; trackBy: trackByFn">
  {{ item.name }}
</div>
```

## Data & Rendering Performance

### Issue: Large Lists Without Virtualization

```typescript
// INEFFICIENT: Renders all 10,000 items in DOM
<div *ngFor="let item of largeList">
  {{ item.name }}
</div>

// OPTIMIZED: Virtual scroll - only renders visible items
<cdk-virtual-scroll-viewport itemSize="50" class="list-viewport">
  <div *cdkVirtualFor="let item of largeList">
    {{ item.name }}
  </div>
</cdk-virtual-scroll-viewport>
```

### Issue: Unnecessary API Calls

```typescript
// INEFFICIENT: API call on every input change
export class SearchComponent {
  constructor(private api: ApiService) {}

  onSearchChange(query: string) {
    this.api.search(query).subscribe(results => {
      this.results = results;
    });
  }
}

// OPTIMIZED: Debounce and cache requests
search$ = this.searchInput$.pipe(
  debounceTime(300),
  distinctUntilChanged(),
  switchMap(query => this.api.search(query)),
  shareReplay(1)
);
```

### Issue: Unoptimized Images

```typescript
// INEFFICIENT: Large images, no optimization
<img src="/assets/large-image.png" width="100">

// OPTIMIZED: Responsive images with srcset
<img 
  src="/assets/image-small.jpg" 
  srcset="/assets/image-medium.jpg 768w, /assets/image-large.jpg 1024w"
  sizes="(max-width: 768px) 100vw, 50vw"
  alt="Description"
>

// OPTIMIZED: Use modern formats
<picture>
  <source srcset="/assets/image.webp" type="image/webp">
  <img src="/assets/image.jpg" alt="Description">
</picture>
```

## Memory Performance

### Issue: Large Objects in Memory

```typescript
// INEFFICIENT: Keeping full data when only display needed
export class DataComponent {
  fullData: LargeObject[] = [];

  ngOnInit() {
    this.api.getData().subscribe(data => {
      this.fullData = data; // Entire object in memory
    });
  }

  get displayData() {
    return this.fullData.slice(0, 10); // Only need 10 items!
  }
}

// OPTIMIZED: Request only needed data
ngOnInit() {
  this.api.getData({ limit: 10 }).subscribe(data => {
    this.displayData = data;
  });
}
```

### Issue: String Concatenation in Loops

```typescript
// INEFFICIENT: Creates new string for each iteration
let result = '';
for (let i = 0; i < items.length; i++) {
  result += items[i].name + ',';
}

// OPTIMIZED: Use array join
const result = items.map(item => item.name).join(',');
```

## Network Performance

### Issue: Multiple Requests When One Would Suffice

```typescript
// INEFFICIENT: Three separate requests
this.api.getUser().subscribe(user => this.user = user);
this.api.getSettings().subscribe(settings => this.settings = settings);
this.api.getPreferences().subscribe(prefs => this.preferences = prefs);

// OPTIMIZED: Single request returning all data
this.api.getUserData().subscribe(data => {
  this.user = data.user;
  this.settings = data.settings;
  this.preferences = data.preferences;
});
```

### Issue: No Request Caching

```typescript
// INEFFICIENT: New request on every component load
ngOnInit() {
  this.api.getConfig().subscribe(config => this.config = config);
}

// OPTIMIZED: Cache results
private configCache$ = this.api.getConfig().pipe(shareReplay(1));

ngOnInit() {
  this.configCache$.subscribe(config => this.config = config);
}
```

## CSS & Layout Performance

### Issue: Expensive CSS Properties

```css
/* INEFFICIENT: box-shadow causes repaints */
.card:hover {
  box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

/* ✅ OPTIMIZED: Use transform instead */
.card:hover {
  transform: translateY(-2px);
}

/* INEFFICIENT: Blurs trigger expensive operations */
.blur-effect {
  filter: blur(10px);
}

/* ✅ OPTIMIZED: Use will-change for browser hint */
.animation-element {
  will-change: transform;
  animation: slide 1s;
}
```

### Issue: Layout Thrashing

```typescript
// INEFFICIENT: Alternating reads/writes causes reflow
for (let i = 0; i < elements.length; i++) {
  elements[i].style.width = elements[i].offsetWidth + 10 + 'px';
}

// OPTIMIZED: Batch reads then writes
const widths = elements.map(el => el.offsetWidth);
elements.forEach((el, i) => {
  el.style.width = widths[i] + 10 + 'px';
});
```

## Component Architecture

### Issue: Monolithic Components

```typescript
// INEFFICIENT: One large component with many inputs/outputs
@Component({
  selector: 'app-user-profile',
  template: `<!-- 500+ lines of template -->`
})
export class UserProfileComponent {
  // Entire profile, posts, comments, etc.
}

// OPTIMIZED: Break into smaller components
// Reuses OnPush change detection isolation
<app-user-header [user]="user"></app-user-header>
<app-user-posts [posts]="posts"></app-user-posts>
<app-user-comments [comments]="comments"></app-user-comments>
```

## Performance Checklist

Before code submission:

- [ ] OnPush change detection used where possible
- [ ] *ngFor uses trackBy function
- [ ] Large lists use virtual scroll (CDK)
- [ ] API requests debounced/throttled appropriately
- [ ] Results cached with shareReplay when suitable
- [ ] No expensive operations in templates
- [ ] Images optimized and responsive (srcset)
- [ ] No layout thrashing (batch DOM reads/writes)
- [ ] Event listeners throttled/debounced
- [ ] Components broken into smaller pieces
- [ ] Subscriptions managed (takeUntil or async pipe)
- [ ] No unnecessary object creation in loops
- [ ] CSS animations use transform/opacity (GPU accelerated)
- [ ] will-change used judiciously for animations
- [ ] No memory leaks from kept references

## Profiling Commands

```bash
# Build for production
ng build --prod

# Analyze bundle size
npm run build -- --stats-json
webpack-bundle-analyzer dist/app/stats.json

# Chrome DevTools Performance tab
- Record
- Do action
- Stop
- Analyze frame rate, main thread, scripting time
```

## Tools for Performance Measurement

- **Chrome DevTools Performance Tab**: Record and analyze CPU profile
- **Lighthouse**: Automated performance audit
- **Angular DevTools**: Chrome extension for Angular-specific profiling
- **WebPageTest**: Detailed performance metrics
- **RxJS DevTools**: Observable performance tracking
