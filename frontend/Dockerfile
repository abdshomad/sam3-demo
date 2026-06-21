FROM node:20-alpine

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package.json package-lock.json ./

# Install dependencies
RUN --mount=type=cache,target=/root/.npm \
    npm install

# Copy the rest of the application
COPY . .

# Expose Next.js port
EXPOSE 3000

# Start production server (build at runtime startup to avoid volume masking)
CMD ["sh", "-c", "npm run build && npm run start"]
