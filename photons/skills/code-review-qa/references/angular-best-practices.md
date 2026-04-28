# Angular Best Practices

Code review guidelines specific to Angular development patterns.

## Component Design

### Single Responsibility Principle

```typescript
// VIOLATES SRP: Component does too much
@Component({
  selector: 'app-user-management',
  template: `...` // 300 lines with UI for list, form, profile, settings
})
export class UserManagementComponent {
  // Handles filtering, sorting, CRUD operations, validation
}

// FOLLOWS SRP: Separate concerns
<app-user-list [users]="users" (onSelect)="selectUser($event)"></app-user-list>
<app-user-form [user]="selectedUser" (onSave)="saveUser($event)"></app-user-form>
<app-user-profile [user]="selectedUser"></app-user-profile>
```

### Smart vs Presentational Components

```typescript
// SMART COMPONENT: Fetches data, manages state
@Component({
  selector: 'app-user-container',
  template: `<app-user-list [users]="users$ | async"></app-user-list>`
})
export class UserContainerComponent {
  users$ = this.userService.getUsers();
  constructor(private userService: UserService) {}
}

// PRESENTATIONAL COMPONENT: Receives data via Input
@Component({
  selector: 'app-user-list',
  template: `<div *ngFor="let user of users">{{ user.name }}</div>`
})
export class UserListComponent {
  @Input() users: User[];
}
```

## Change Detection

### OnPush Strategy

```typescript
// DEFAULT: Checks on every async action
@Component({
  selector: 'app-item'
})
export class ItemComponent {
  item: Item;
}

// OPTIMIZED: Only checks when @Input changes
@Component({
  selector: 'app-item',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class ItemComponent {
  @Input() item: Item;
}
```

### Template Expressions

```typescript
// INEFFICIENT: Called on every cycle
<div>{{ complexMethod() }}</div>

// OPTIMIZED: Cached result
computed = this.complexMethod();
// Template
<div>{{ computed }}</div>

// OPTIMIZED: Use pipe
<div>{{ value | customPipe }}</div>
```

## Dependency Injection

### Service Scope Management

```typescript
// WRONG: Service provided in component (new instance per component)
@Component({
  providers: [UserService]
})
export class UserComponent {
  constructor(private userService: UserService) {}
}

// CORRECT: Service provided in module (singleton)
@NgModule({
  providers: [UserService]
})

// CORRECT: Service scope managed appropriately
@Component({
  // Component-scoped service if isolation needed
  providers: [FormService]
})
```

### Constructor Injection

```typescript
// WRONG: Manual service creation
export class UserComponent {
  userService = new UserService();
}

// CORRECT: Dependency injection
export class UserComponent {
  constructor(private userService: UserService) {}
}
```

## Observable Patterns

### Subscribe vs Async Pipe

```typescript
// MANUAL SUBSCRIPTION: Memory leak risk
export class UserComponent implements OnInit {
  user: User;

  ngOnInit() {
    this.userService.getUser().subscribe(user => {
      this.user = user;
    });
  }
  // No unsubscribe!
}

// ASYNC PIPE: Automatic cleanup
export class UserComponent {
  user$ = this.userService.getUser();
}
// Template: {{ user$ | async }}

// TAKEUNTIL PATTERN: Explicit cleanup
private destroy$ = new Subject<void>();

ngOnInit() {
  this.userService.getUser()
    .pipe(takeUntil(this.destroy$))
    .subscribe(user => this.user = user);
}

ngOnDestroy() {
  this.destroy$.next();
  this.destroy$.complete();
}
```

### Observable Composition

```typescript
// NESTED SUBSCRIPTIONS: "Subscription Hell"
this.service1.get().subscribe(data1 => {
  this.service2.get(data1).subscribe(data2 => {
    this.service3.get(data2).subscribe(data3 => {
      this.data = data3;
    });
  });
});

// FLAT MAP OPERATORS: Composed observables
this.data$ = this.service1.get().pipe(
  switchMap(data1 => this.service2.get(data1)),
  switchMap(data2 => this.service3.get(data2))
);

// COMBINE OBSERVABLES: Parallel requests
this.data$ = combineLatest([
  this.service1.get(),
  this.service2.get(),
  this.service3.get()
]).pipe(
  map(([data1, data2, data3]) => ({ data1, data2, data3 }))
);
```

## Forms

### Reactive Forms Best Practice

```typescript
// TEMPLATE-DRIVEN: Limited validation control
<form #form="ngForm">
  <input [(ngModel)]="user.name" required>
</form>

// REACTIVE FORMS: Better control & testing
export class UserForm {
  form = new FormGroup({
    name: new FormControl('', Validators.required),
    email: new FormControl('', [Validators.required, Validators.email])
  });
}

// Template
<form [formGroup]="form" (ngSubmit)="onSubmit()">
  <input formControlName="name">
  <div *ngIf="form.get('name')?.hasError('required')">Required</div>
</form>
```

### Form Validation

```typescript
// CLEAR ERROR HANDLING
get emailError(): string {
  const control = this.form.get('email');
  if (!control || !control.errors) return '';
  
  if (control.hasError('required')) return 'Email is required';
  if (control.hasError('email')) return 'Email format invalid';
  return '';
}

// Template
<div *ngIf="emailError">{{ emailError }}</div>
```

## Routing

### Lazy Loading

```typescript
// LAZY LOAD: Route module loaded only when accessed
const routes: Routes = [
  {
    path: 'admin',
    loadChildren: () => import('./admin/admin.module').then(m => m.AdminModule)
  }
];

// PRELOAD STRATEGY: Load background routes
@NgModule({
  imports: [RouterModule.forRoot(routes, {
    preloadingStrategy: PreloadAllModules
  })]
})
```

### Route Guards

```typescript
// PROTECT ROUTES: Check before navigation
@Injectable()
export class AuthGuard implements CanActivate {
  constructor(private auth: AuthService, private router: Router) {}

  canActivate(): Observable<boolean> {
    return this.auth.isAuthenticated().pipe(
      map(isAuth => isAuth || this.router.parseUrl('/login'))
    );
  }
}

// Use in routes
const routes: Routes = [
  {
    path: 'admin',
    component: AdminComponent,
    canActivate: [AuthGuard]
  }
];
```

## HTTP & API Integration

### HTTP Error Handling

```typescript
// UNHANDLED ERRORS: Silent failures
this.http.get('/api/users').subscribe(
  data => this.users = data
  // No error handler!
);

// PROPER ERROR HANDLING:
this.http.get('/api/users').pipe(
  catchError(error => {
    console.error('Failed to load users:', error);
    this.showErrorMessage('Unable to load users');
    return of([]);  // Return empty array as fallback
  })
).subscribe(data => this.users = data);
```

### HTTP Interceptors

```typescript
// CENTRALIZED: Auth token in all requests
@Injectable()
export class AuthInterceptor implements HttpInterceptor {
  intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    const token = this.auth.getToken();
    req = req.clone({
      setHeaders: { Authorization: `Bearer ${token}` }
    });
    return next.handle(req);
  }
}

// Register in providers
providers: [
  { provide: HTTP_INTERCEPTORS, useClass: AuthInterceptor, multi: true }
]
```

## Directives & Pipes

### Custom Directives

```typescript
// REUSABLE: Attribute directive for highlighting
@Directive({
  selector: '[appHighlight]'
})
export class HighlightDirective {
  constructor(private el: ElementRef, private renderer: Renderer2) {}

  @HostListener('mouseenter') onEnter() {
    this.renderer.setStyle(this.el.nativeElement, 'backgroundColor', 'yellow');
  }

  @HostListener('mouseleave') onLeave() {
    this.renderer.setStyle(this.el.nativeElement, 'backgroundColor', 'transparent');
  }
}

// Usage: <span appHighlight>Highlighted text</span>
```

### Pure Pipes

```typescript
// OPTIMIZED: Pure pipe cached by Angular
@Pipe({
  name: 'currency',
  pure: true  // Default
})
export class CurrencyPipe implements PipeTransform {
  transform(value: number): string {
    return `$${value.toFixed(2)}`;
  }
}

// IMPURE: Called every change detection
@Pipe({
  name: 'impure',
  pure: false
})
export class ImpurePipe {
  // Avoid unless absolutely necessary
}
```

## Module Organization

### Feature Module Structure

```typescript
// WELL-ORGANIZED: Feature encapsulation
@NgModule({
  imports: [CommonModule, SharedModule],
  declarations: [UserListComponent, UserDetailComponent],
  providers: [UserService]
})
export class UserModule {}

// WRONG: Exposing all declarations
exports: [UserListComponent, UserDetailComponent, UserService]

// CORRECT: Export only if reused outside module
exports: [SharedComponent]
```

### Shared Module

```typescript
// SHARED MODULE: Common declarations & imports
@NgModule({
  declarations: [HighlightDirective, CurrencyPipe],
  imports: [CommonModule, FormsModule],
  exports: [HighlightDirective, CurrencyPipe, CommonModule, FormsModule]
})
export class SharedModule {}

// Use in other modules
@NgModule({
  imports: [SharedModule]
})
export class FeatureModule {}
```

## Best Practices Checklist

Before code submission:

**Architecture**  

- [ ] Single responsibility principle followed
- [ ] Smart vs presentational components properly separated
- [ ] Services handle logic, components handle presentation
- [ ] DI used correctly (no manual instantiation)

**Performance**  

- [ ] OnPush change detection used where applicable
- [ ] *ngFor uses trackBy function
- [ ] No memory leaks (takeUntil pattern)
- [ ] No heavy computations in templates

**Observables**  

- [ ] All subscriptions properly cleaned up
- [ ] Observable composition used (not nested subscriptions)
- [ ] Async pipe used in templates when possible
- [ ] Error handling present

**Forms**  

- [ ] Reactive forms used for complex validation
- [ ] Clear error messages
- [ ] Proper form state management

**Routing**  

- [ ] Lazy loading implemented where beneficial
- [ ] Route guards protect sensitive routes
- [ ] 404 handling present

**HTTP**  

- [ ] API errors handled gracefully
- [ ] HTTP interceptors for cross-cutting concerns
- [ ] Timeout and retry logic as needed

**Code Quality**  

- [ ] Type safety enforced (no any types)
- [ ] Consistent naming conventions
- [ ] Comments for complex logic
- [ ] Tests cover critical paths
