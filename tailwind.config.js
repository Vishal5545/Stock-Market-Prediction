module.exports = {
  theme: {
    extend: {
      colors: {
        primary: {
          light: '#4B83DB',
          DEFAULT: '#1e3c72',
          dark: '#0c2d5f'
        },
        secondary: {
          light: '#4B9EDB',
          DEFAULT: '#2a5298',
          dark: '#1a3f7f'
        },
        success: {
          light: '#10b981',
          DEFAULT: '#059669',
          dark: '#047857'
        },
        error: {
          light: '#ef4444',
          DEFAULT: '#dc2626',
          dark: '#b91c1c'
        },
        warning: {
          light: '#f59e0b',
          DEFAULT: '#d97706',
          dark: '#b45309'
        },
        info: {
          light: '#3b82f6',
          DEFAULT: '#2563eb',
          dark: '#1d4ed8'
        },
        gray: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a'
        }
      },
      spacing: {
        '128': '32rem',
        '144': '36rem'
      },
      borderRadius: {
        '4xl': '2rem'
      },
      fontFamily: {
        sans: ['Inter var', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif']
      },
      boxShadow: {
        sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        DEFAULT: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1)',
        md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1)',
        lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1)',
        xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)',
        '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
        inner: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.05)'
      }
    }
  },
  variants: {
    extend: {
      backgroundColor: ['active', 'group-hover'],
      textColor: ['group-hover'],
      transform: ['group-hover'],
      scale: ['group-hover'],
      translate: ['group-hover']
    }
  }
}