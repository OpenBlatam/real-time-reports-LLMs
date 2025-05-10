# Makefile (root)

.PHONY: dev api-dev-fg web-dev-fg stop

# Main development target: runs both API and Web dev servers in parallel.
# They will run until Ctrl+C is pressed.
dev:
	@echo "Starting API and Web development servers..."
	@echo "Press Ctrl+C to stop all."
	# Start API dev server in the background
	# The subshell ( ... ) ensures that 'cd' doesn't affect subsequent commands at this level.
	(cd api && $(MAKE) dev) & \
	# Start Web dev server in the background
	(cd web && $(MAKE) dev) & \
	# Wait for all background jobs of this shell to complete.
	# Since dev servers run indefinitely, this 'wait' will also run indefinitely until interrupted (Ctrl+C).
	wait
	@echo "Development servers stopped or shell command finished."

# Target to stop API-specific services (like Docker containers, Supabase).
# The web dev server (npm run dev) is expected to be stopped when 'make dev' is interrupted (Ctrl+C).
stop:
	@echo "Stopping API services (Redis, Supabase)..."
	$(MAKE) -C api stop
	@echo "API services stopped. If 'make dev' was running, web server should also be stopped."

# Individual targets if you want to run them separately (e.g., in different terminal tabs)
# These are foreground targets.
api-dev-fg:
	@echo "Starting API development server (foreground)..."
	cd api && $(MAKE) dev

web-dev-fg:
	@echo "Starting Web development server (foreground)..."
	cd web && $(MAKE) dev 