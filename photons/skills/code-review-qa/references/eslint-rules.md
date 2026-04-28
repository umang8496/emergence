# ESLint Rules Reference

Configure and enforce code quality rules automatically during development.

## Recommended ESLint Configuration

```json
// .eslintrc.json
{
  "extends": [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:@angular-eslint/recommended"
  ],
  "rules": {
    "no-console": ["warn", { "allow": ["warn", "error"] }],
    "no-debugger": "error",
    "no-unused-vars": "off",  // Use TS version instead
    "@typescript-eslint/no-unused-vars": ["error", {
      "argsIgnorePattern": "^_"
    }],
    "@typescript-eslint/no-explicit-any": "error",
    "@typescript-eslint/explicit-function-return-types": "warn",
    "@typescript-eslint/no-non-null-assertion": "warn",
    "no-var": "error",
    "prefer-const": "error",
    "eqeqeq": ["error", "always"],
    "curly": "error"
  }
}
```

## Critical Rules (Must Fix)

| Rule                               | Issue                         | Example                          |
|------------------------------------|-------------------------------|----------------------------------|
| no-debugger                        | Left-over debugger statements | debugger;                        |
| no-console (error)                 | Production console logs       | console.log('test') in prod code |
| @typescript-eslint/no-explicit-any | Unsafe type                   | const x: any = data              |
| no-eval                            | Security vulnerability        | eval(userInput)                  |
| no-implied-eval                    | Indirect eval                 | setTimeout('code', 1000)         |
| no-new-func                        | Function constructor security | new Function('code')()           |

## Important Rules (Should Fix)

| Rule           | Issue                             | Example                  | Fix                    |
|----------------|-----------------------------------|--------------------------|------------------------|
| no-unused-vars | Dead code                         | let x = 5; // never used | Remove or use variable |
| prefer-const   | Mutable when immutable sufficient | let x = 5;               | const x = 5;           |
| eqeqeq         | Type coercion bugs                | if (x == 5)              | if (x === 5)           |
| no-var         | Scope confusion                   | var x = 5;               | const x = 5;           |
| curly          | Missing braces                    | if (x) y();              | if (x) { y(); }        |

## TypeScript-Specific Rules

```json
{
  "rules": {
    // Type safety
    "@typescript-eslint/no-explicit-any": "error",
    "@typescript-eslint/no-non-null-assertion": "warn",
    "@typescript-eslint/strict-boolean-expressions": "warn",
    
    // Unused code
    "@typescript-eslint/no-unused-vars": ["error", {
      "argsIgnorePattern": "^_|^__"
    }],
    
    // Naming conventions
    "@typescript-eslint/naming-convention": ["warn", {
      "selector": "default",
      "format": ["camelCase"]
    }],
    
    // Function types
    "@typescript-eslint/explicit-function-return-types": ["warn", {
      "allowExpressions": true
    }],
    
    // Null handling
    "@typescript-eslint/no-non-null-asserted-optional-chain": "error"
  }
}
```

## Angular-Specific Rules

```json
{
  "rules": {
    "@angular-eslint/directive-selector": ["error", {
      "type": "attribute",
      "prefix": "app",
      "style": "camelCase"
    }],
    "@angular-eslint/component-selector": ["error", {
      "type": "element",
      "prefix": "app",
      "style": "kebab-case"
    }],
    "@angular-eslint/no-empty-lifecycle-method": "error",
    "@angular-eslint/no-host-metadata-property": "error",
    "@angular-eslint/use-lifecycle-interface": "error",
    "@angular-eslint/use-pipe-transform-interface": "error"
  }
}
```

## Performance Rules

```json
{
  "rules": {
    // Avoid inefficient operations
    "no-unneeded-ternary": "warn",
    "no-nested-ternary": "warn",
    
    // Alert to performance issues
    "complexity": ["warn", 10],  // Cyclomatic complexity
    "max-depth": ["warn", 4],    // Nesting depth
    "max-lines-per-function": ["warn", 100],
    
    // Loops and operations
    "no-loop-func": "error",
    "no-param-reassign": "warn"
  }
}
```

## Style Rules

```json
{
  "rules": {
    // Consistency
    "indent": ["error", 2],
    "quotes": ["error", "single", { "avoidEscape": true }],
    "semi": ["error", "always"],
    "comma-dangle": ["error", "never"],
    
    // Spacing
    "space-before-function-paren": ["error", {
      "anonymous": "always",
      "named": "never",
      "asyncArrow": "always"
    }],
    "object-curly-spacing": ["error", "always"],
    "array-bracket-spacing": ["error", "never"],
    
    // Readability
    "max-len": ["warn", { "code": 100 }],
    "no-multiple-empty-lines": ["error", { "max": 2 }]
  }
}
```

## Running ESLint

```bash
# Check all files
npm run lint

# Check specific file
npm run lint -- src/app/component/my.component.ts

# Fix automatically where possible
npm run lint -- --fix

# Fix specific file
npm run lint -- --fix src/app/component/my.component.ts

# Generate report
npm run lint -- --format json > lint-report.json
```

## Pre-commit Hook Integration

```json
// package.json
{
  "husky": {
    "hooks": {
      "pre-commit": "lint-staged"
    }
  },
  "lint-staged": {
    "src/**/*.ts": ["eslint --fix", "git add"],
    "src/**/*.{html,scss}": ["prettier --write", "git add"]
  }
}
```

## IDE Integration

### VS Code Settings

```json
// .vscode/settings.json
{
  "eslint.enable": true,
  "eslint.validate": [
    "typescript",
    "typescriptreact"
  ],
  "[typescript]": {
    "editor.defaultFormatter": "dbaeumer.vscode-eslint",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll.eslint": true
    }
  }
}
```

## Rules Severity Levels

| Severity           | Action                              |
|--------------------|-------------------------------------|
| error              | Must fix before merge               |
| warn               | Fix recommended                     |
| off                | Disabled (not checked)              |

## Common Overrides for Team

```json
{
  "rules": {
    // Stricter for production code
    "no-console": ["error", { "allow": [] }],
    
    // Allow prefixed unused params (e.g., _unused)
    "@typescript-eslint/no-unused-vars": ["error", {
      "argsIgnorePattern": "^_"
    }],
    
    // Require explicit return types
    "@typescript-eslint/explicit-function-return-types": "error",
    
    // Warn on any types, but allow with comment
    "@typescript-eslint/no-explicit-any": ["warn", {
      "fixToUnknown": false
    }]
  },
  "overrides": [
    {
      // Less strict for test files
      "files": ["**/*.spec.ts"],
      "rules": {
        "@typescript-eslint/no-explicit-any": "off",
        "no-console": "off"
      }
    }
  ]
}
```

## Troubleshooting ESLint

### Rule Not Working

1. Check syntax in `.eslintrc.json`
2. Ensure plugin installed: `npm install --save-dev eslint-plugin-name`
3. Verify rule name in plugin docs
4. Restart IDE/linter: `npm run lint -- --fix`

### Too Many Warnings

- Review default rule severity
- Override rules in `.eslintrc.json`
- Use overrides for specific file types/folders
- Document exceptions with `// eslint-disable-next-line rule-name`

### Performance Issues

- Cache results: `--cache` flag
- Lint only changed files in CI
- Run incrementally during development
- Use pre-commit hooks to catch issues early

## Resources

- [ESLint Docs](https://eslint.org/docs/rules/)
- [@typescript-eslint Rules](https://github.com/typescript-eslint/typescript-eslint/tree/main/packages/eslint-plugin/docs/rules)
- [@angular-eslint Rules](https://github.com/angular-eslint/angular-eslint#usage)
