# Pocketpal

**Your walking companion for solo travel.**

Pocketpal is an open-source app that makes walking alone feel less lonely and more enriching.  
It helps you plan short walking routes, guides you along the way, and tells you stories about the places you pass.  
Think of it as a *walking encyclopedia in your pocket*.

---

## Why Pocketpal?

Travelling solo can be exciting, but it can also feel boring, lonely, or even intimidating.  
Tour groups are often rigid and expensive, while wandering alone may leave you missing the history and stories behind the places you pass.

Pocketpal is for solo travellers who want freedom and flexibility, but still want company, cultural insights, and a sense of safety.

**Example prompts you can give the app:**
- "I’d like a 30-minute green walk from the station"
- "I want to go shopping for an hour in Shibuya"
- "Show me the most cultural route near King’s Cross"

The app responds with a loop route, highlights interesting places, and narrates short stories about them hands-free as you walk.

---

## How it works (MVP)

### On your phone (frontend)
- Displays a map using MapLibre  
- Lets you type a natural-language request  
- Tracks your GPS location as you walk  
- Plays narration when you’re near a point of interest  

### On the server (backend, built in Python with FastAPI)
- **Intent Parser**: interprets your request and figures out what matters most (green spaces, culture, shopping, etc.)  
- **Route Generator**: calls a routing service (like OpenRouteService or OSRM) to create loop routes that fit your time limit  
- **Scorers**: measure different qualities of the route:
  - Greenness: how much of the route passes through parks or green areas
  - Coolness: shade and shelter, useful in hot weather
  - Culture: number and importance of landmarks along the way
  - Detour penalty: whether the route fits the requested duration
- **Arbiter**: chooses the best route based on these scores  
- **Narrator**: generates short blurbs for each landmark (using cached language model output)  

### Data sources
- OpenStreetMap (via Overpass API) for green areas and POIs  
- Wikidata/Wikipedia for cultural information  
- Weather API for temperature and heat index  

---

## First GIS feature

Pocketpal includes a **Coolness score**: how much of a route is near green or sheltered areas.  

When it’s hot (for example, 30 °C or higher), the app automatically shortens suggested walks and prefers shaded or indoor routes.

---

## What users see

- A map with a walking route tailored to their request  
- Three to five points of interest along the way  
- Friendly audio blurbs (100–150 words each) played automatically as they approach a landmark  
- Clear explanations of why a route was chosen  
  - *Example*: "This route is 62% near green areas and passes 2 indoor POIs, so it should feel cooler today."

---

## Roadmap

- **v0.1**: Simple loop routes with greenness and narration  
- **v0.2**: Shopping and cultural routes with smarter POI selection  
- **v0.3**: Offline packs for popular cities  
- **v0.4**: Optional buddy finder for solo travellers  

---

## License

Pocketpal is open source and released under the MIT license.  
Contributions are welcome.
