# Memory Leak Patterns & Prevention

Identify and eliminate memory leaks before they cause performance degradation.

## Observable Subscriptions

### Pattern 1: Unmanaged Subscriptions

```typescript
// MEMORY LEAK
export class UserComponent implements OnInit {
  constructor(private userService: UserService) {}

  ngOnInit() {
    this.userService.getUser().subscribe(user => {
      this.user = user;
    }); // No unsubscribe! Stays in memory when component destroyed
  }
}

// FIXED: Using takeUntil pattern
export class UserComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();

  constructor(private userService: UserService) {}

  ngOnInit() {
    this.userService.getUser()
      .pipe(takeUntil(this.destroy$))
      .subscribe(user => {
        this.user = user;
      });
  }

  ngOnDestroy() {
    this.destroy$.next();
    this.destroy$.complete();
  }
}

// FIXED: Using async pipe (no unsubscribe needed)
export class UserComponent {
  user$ = this.userService.getUser();

  constructor(private userService: UserService) {}
}
// Template: {{ user$ | async }}
```

### Pattern 2: Route Parameter Subscriptions

```typescript
// MEMORY LEAK
ngOnInit() {
  this.route.params.subscribe(params => {
    this.id = params.id;
    this.loadData();
  });
}

// FIXED
private destroy$ = new Subject<void>();

ngOnInit() {
  this.route.params
    .pipe(takeUntil(this.destroy$))
    .subscribe(params => {
      this.id = params.id;
      this.loadData();
    });
}

ngOnDestroy() {
  this.destroy$.next();
  this.destroy$.complete();
}
```

### Pattern 3: Query Parameters Subscriptions

```typescript
// MEMORY LEAK
ngOnInit() {
  this.route.queryParams.subscribe(params => {
    this.search = params.q;
  });
}

// FIXED
ngOnInit() {
  this.route.queryParams
    .pipe(takeUntil(this.destroy$))
    .subscribe(params => {
      this.search = params.q;
    });
}
```

## Event Listeners

### Pattern 4: Window & Document Events

```typescript
// MEMORY LEAK
ngOnInit() {
  window.addEventListener('resize', this.onResize);
}

// FIXED: Remove listener in ngOnDestroy
ngOnInit() {
  window.addEventListener('resize', this.onResize);
}

ngOnDestroy() {
  window.removeEventListener('resize', this.onResize);
}

// FIXED: Use RxJS fromEvent
ngOnInit() {
  fromEvent(window, 'resize')
    .pipe(
      throttleTime(300),
      takeUntil(this.destroy$)
    )
    .subscribe(() => this.onResize());
}
```

### Pattern 5: HTML Element Event Listeners

```typescript
// MEMORY LEAK
ngOnInit() {
  this.button.addEventListener('click', this.onClick);
}

// FIXED: Remove listener in ngOnDestroy
ngOnInit() {
  this.button.addEventListener('click', this.onClick);
}

ngOnDestroy() {
  this.button.removeEventListener('click', this.onClick);
}

// FIXED: Use Renderer2
constructor(private renderer: Renderer2) {}

ngOnInit() {
  this.unlisten = this.renderer.listen(this.button, 'click', () => {
    this.onClick();
  });
}

ngOnDestroy() {
  this.unlisten(); // Renderer2 returns unlistener function
}
```

## Timers

### Pattern 6: SetTimeout & SetInterval

```typescript
// MEMORY LEAK
ngOnInit() {
  this.timerId = setTimeout(() => {
    this.loadData();
  }, 5000);
}
// Timer still fires even after component destroyed!

// FIXED: Clear timer in ngOnDestroy
ngOnDestroy() {
  clearTimeout(this.timerId);
}

// MEMORY LEAK: Interval
ngOnInit() {
  this.intervalId = setInterval(() => {
    this.refreshData();
  }, 10000);
}

// FIXED: Clear interval
ngOnDestroy() {
  clearInterval(this.intervalId);
}

// FIXED: Use RxJS interval
ngOnInit() {
  interval(10000)
    .pipe(takeUntil(this.destroy$))
    .subscribe(() => this.refreshData());
}
```

## Component References

### Pattern 7: Keeping References to Components

```typescript
// MEMORY LEAK
export class ParentComponent {
  children: ChildComponent[] = [];

  addChild() {
    this.children.push(new ChildComponent()); // Holds reference
  }

  removeChild(index: number) {
    this.children.splice(index, 1); // Child component still referenced!
  }
}

// FIXED: Properly cleanup references
removeChild(index: number) {
  const child = this.children[index];
  child.ngOnDestroy?.();  // Manually trigger cleanup if needed
  this.children.splice(index, 1);
}

// FIXED: Use Angular's built-in cleanup
// Let Angular handle component destruction through ngIf or *ngFor
<ng-container *ngFor="let child of children">
  <app-child [data]="child"></app-child>
</ng-container>
```

## Third-Party Libraries

### Pattern 8: Library Cleanup

```typescript
// MEMORY LEAK: Chart.js not destroyed
ngOnInit() {
  this.chart = new Chart(this.canvasRef.nativeElement, config);
}

// FIXED: Destroy chart on component destroy
ngOnDestroy() {
  if (this.chart) {
    this.chart.destroy();
  }
}

// MEMORY LEAK: Google Maps not cleaned up
ngOnInit() {
  this.map = new google.maps.Map(this.mapRef.nativeElement, options);
}

// FIXED: Clean up map
ngOnDestroy() {
  if (this.map) {
    google.maps.event.clearInstanceListeners(this.map);
  }
}
```

## Detection Patterns

### How to Identify Memory Leaks

1. **Browser DevTools - Memory Tab**
   - Take heap snapshots before and after navigation
   - Look for retained component instances
   - Check for orphaned event listeners

2. **Observable Leaks**
   - Search for `.subscribe(` without `takeUntil`
   - Look for route/query subscriptions in ngOnInit
   - Check for missing ngOnDestroy

3. **Event Listener Leaks**
   - Search for `addEventListener` without matching `removeEventListener`
   - Look for uncanceled Renderer2 listeners
   - Check for jQuery/third-party event bindings

4. **Timer Leaks**
   - Search for `setTimeout`/`setInterval` without clear calls
   - Look for missing timer cleanup in ngOnDestroy

## Best Practices

### Use the Destroy Subject Pattern

```typescript
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';

export class BaseComponent implements OnDestroy {
  protected destroy$ = new Subject<void>();

  ngOnDestroy() {
    this.destroy$.next();
    this.destroy$.complete();
  }
}

// All subscriptions
this.service.method()
  .pipe(takeUntil(this.destroy$))
  .subscribe();
```

### Use Async Pipe in Templates

```typescript
// Instead of manual subscriptions
export class MyComponent {
  data$ = this.service.getData();
}

// Template
<div>{{ data$ | async }}</div>
```

### Use Unsubscribe in Subscription Variable

```typescript
private sub: Subscription;

ngOnInit() {
  this.sub = this.service.getData().subscribe(data => {
    this.data = data;
  });
}

ngOnDestroy() {
  this.sub.unsubscribe();
}
```

## Memory Leak Checklist

Before submitting code:

- [ ] All `.subscribe()` calls use `takeUntil(destroy$)`
- [ ] `ngOnDestroy` implemented if component has subscriptions/listeners
- [ ] All `addEventListener` calls have matching `removeEventListener`
- [ ] All `setTimeout`/`setInterval` calls are cleared in `ngOnDestroy`
- [ ] All third-party library instances are properly destroyed
- [ ] No global references to component instances
- [ ] Event listeners use Renderer2 (proper Angular way)
- [ ] Form subscriptions properly cleaned up
- [ ] Route/query subscriptions use takeUntil pattern
- [ ] No circular references between components
