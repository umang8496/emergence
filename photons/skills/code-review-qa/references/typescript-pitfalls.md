# Common TypeScript Pitfalls in Angular

Track down and prevent these common TypeScript issues before they cause runtime failures.

## Null & Undefined Handling

### Pitfall 1: Direct Property Access Without Guards

```typescript
// DANGEROUS - If user is null/undefined, app crashes
const email = user.profile.email;

// SAFE - Optional chaining operator
const email = user?.profile?.email;

// SAFE - With default fallback
const email = user?.profile?.email ?? 'no-email@example.com';
```

**Prevention:**  

- Enable TypeScript strict null checks in `tsconfig.json`:

  ```json
  "strict": true,
  "strictNullChecks": true
  ```

- Use optional chaining (`?.`)
- Use nullish coalescing (`??`) for defaults

### Pitfall 2: Array Access Without Bounds Check

```typescript
// DANGEROUS - What if array is empty?
const firstItem = items[0].id;

// SAFE - Check length first
const firstItem = items.length > 0 ? items[0].id : null;

// SAFE - Optional chaining
const firstItem = items[0]?.id;
```

### Pitfall 3: API Response Assumptions

```typescript
// DANGEROUS - Assuming response structure
this.api.getData().subscribe(data => {
  this.title = data.result.title; // What if data.result is null?
});

// SAFE - Guard against missing properties
this.api.getData().subscribe(data => {
  this.title = data?.result?.title ?? 'Untitled';
  if (!data?.result?.title) {
    console.warn('Missing title in response');
  }
});
```

### Pitfall 4: Form Control Value Handling

```typescript
// DANGEROUS - Value might be null
const email = this.form.get('email').value;

// SAFE - Handle null control
const email = this.form.get('email')?.value;

// SAFE - Type-safe form access
const email = this.form.get('email')?.value?.trim() ?? '';
```

## Async & Promise Handling

### Pitfall 5: Floating Promises (Fire and Forget)

```typescript
// DANGEROUS - Promise not awaited, errors ignored
this.dataService.save(data);  // What if it fails?

// SAFE - Properly subscribe and handle
this.dataService.save(data).subscribe(
  () => { /* success */ },
  error => { /* handle error */ }
);

// SAFE - With error logging
this.dataService.save(data).subscribe(
  () => { /* success */ },
  error => { 
    console.error('Save failed:', error);
    this.showErrorMessage(error.message);
  }
);
```

### Pitfall 6: Unhandled Promise Rejections

```typescript
// DANGEROUS - No error handler
async loadData() {
  const data = await this.service.fetch();  // What if it throws?
  this.data = data;
}

// SAFE - Try-catch with error handling
async loadData() {
  try {
    const data = await this.service.fetch();
    this.data = data;
  } catch (error) {
    console.error('Failed to load data:', error);
    this.showErrorMessage('Unable to load data');
  }
}
```

## Observable & Subscription Management

### Pitfall 7: Memory Leaks from Unsubscribed Observables

```typescript
// MEMORY LEAK - Observable never unsubscribed
ngOnInit() {
  this.route.params.subscribe(params => {
    this.load(params.id);
  });
  // When component destroys, subscription still active!
}

// SAFE - Using takeUntil pattern
private destroy$ = new Subject<void>();

ngOnInit() {
  this.route.params
    .pipe(takeUntil(this.destroy$))
    .subscribe(params => {
      this.load(params.id);
    });
}

ngOnDestroy() {
  this.destroy$.next();
  this.destroy$.complete();
}
```

### Pitfall 8: Multiple Subscriptions to Same Observable

```typescript
// INEFFICIENT - Creates multiple subscriptions
render() {
  this.data$.subscribe(data => this.render1(data));
  this.data$.subscribe(data => this.render2(data));
}

// SAFE - Single subscription with shareReplay
private data$ = this.service.getData().pipe(shareReplay(1));

ngOnInit() {
  this.data$.subscribe(data => this.render1(data));
  this.data$.subscribe(data => this.render2(data));
}
```

## Type Safety

### Pitfall 9: Using `any` Type

```typescript
// TYPE UNSAFE - Loses all type checking
const data: any = response;
console.log(data.someProperty.nested); // No type checking!

// TYPE SAFE - Define proper interface
interface DataResponse {
  someProperty?: {
    nested?: string;
  };
}
const data: DataResponse = response;
console.log(data.someProperty?.nested); // Type checked!
```

### Pitfall 10: Unsafe Type Casts

```typescript
// DANGEROUS - Cast without verification
const user = response as User;  // What if it's not actually a User?

// SAFE - Type guard before cast
function isUser(obj: any): obj is User {
  return obj && typeof obj.id === 'number' && typeof obj.email === 'string';
}

if (isUser(response)) {
  const user: User = response;  // Now safe to cast
}
```

### Pitfall 11: Missing Return Types

```typescript
// TYPE UNSAFE - Return type inferred
function processData(data) {  // What type is returned?
  return data.transform();
}

// TYPE SAFE - Explicit return type
function processData(data: DataModel): TransformedData {
  return data.transform();
}

// TYPE SAFE - For components
getData(): Observable<User[]> {
  return this.http.get<User[]>('/api/users');
}
```

## State Management Issues

### Pitfall 12: Mutable State Updates

```typescript
// DANGEROUS - Direct mutation
items.push(newItem);  // Doesn't trigger change detection in OnPush

// SAFE - Create new array
items = [...items, newItem];

// SAFE - Or use immutable patterns
items = items.concat(newItem);
```

### Pitfall 13: Shared State Mutations

```typescript
// DANGEROUS - Modifying shared object
const newData = this.sharedData;  // Reference, not copy
newData.property = 'changed';     // Affects shared object!

// SAFE - Create new object
const newData = { ...this.sharedData, property: 'changed' };
```

## Common Patterns to Watch

| Issue                    | Example          | Solution                                |
|--------------------------|------------------|-----------------------------------------|
| Null access              | obj.prop.nested  | Use optional chaining obj?.prop?.nested |
| Unsubscribed observables | obs.subscribe()  | Use takeUntil(destroy$) pattern         |
| Async errors             | promise.then()   | Add .catch() or use try-catch           |
| Any type                 | const x: any     | Replace with specific type              |
| Array mutation           | arr.push(x)      | Use spread [...arr, x]                  |
| Direct state mutation    | obj.prop = value | Create new object {...obj, prop: value} |

## Quick Checklist

Before submitting code, check:

- [ ] All object accesses use optional chaining (`?.`) or null checks
- [ ] All async operations have error handlers
- [ ] All subscriptions are cleaned up (takeUntil, unsubscribe)
- [ ] No `any` types without justification
- [ ] All array accesses check bounds
- [ ] Form controls use safe access (`.get()?.value`)
- [ ] API responses have fallbacks for missing properties
- [ ] No state mutations (use spread operator or Object.assign)
- [ ] All functions have return type annotations
- [ ] Strict null checks enabled in tsconfig.json
