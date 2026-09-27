from .cli import main

# Guarded: engine trunks start workers with the "spawn" method, which re-imports this module.
if __name__ == "__main__":
    main()
