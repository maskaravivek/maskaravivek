+++
# Slider widget.
widget = "slider"  # See https://sourcethemes.com/academic/docs/page-builder/
headless = true  # This file represents a page section.
active = true  # Activate this widget? true/false
weight = 1  # Order that this section will appear.

# Slide interval.
# Use `false` to disable animation or enter a time in ms, e.g. `5000` (5s).
interval = false

# Slide height (optional).
# E.g. `500px` for 500 pixels or `calc(100vh - 70px)` for full screen.
height = ""

# Slides.
# Duplicate an `[[item]]` block to add more slides.
[[item]]
  title = "AI-Powered Markdown Editor"
  content = "Built an AI-powered markdown editor using Next.js, AWS CDK, Lamdba, and DynamoDB for tech writers with intelligent suggestions, code highlighting, and seamless publishing workflows<br><br>"
  align = "center"

  overlay_color = "#2E86AB"  # Professional blue
  overlay_filter = 0.4  # Darken image so text is readable

  cta_label = "Visit mdedit.ai"
  cta_url = "https://mdedit.ai"
  cta_icon_pack = "fas"
  cta_icon = "edit"

[[item]]
  title = "Grain Measurement System"
  content = "Built the cloud infrastructure, mobile app, ML labeling platform, and Stripe payment integration for **Inweon GRAMS**, a secure AI‑powered grain quality analyzer using computer vision and ML to assess rice, wheat, pulses, and oilseeds with ~99% accuracy.<br><br>"
  align = "center"

  overlay_color = "#4C8C2B"  # Agritech green
  overlay_filter = 0.4  # Darken background for readability

  cta_label = "Check out GRAMS"
  cta_url = "https://inweon.com"
  cta_icon_pack = "fas"
  cta_icon = "leaf"

[[item]]
  title = "Android Development Masterclass"
  content = "Created comprehensive Android course for Educative with 60+ lessons and 40+ interactive example apps teaching modern development practices<br><br>"
  align = "center"

  overlay_color = "#3DDC84"  # Android green
  overlay_img = ""
  overlay_filter = 0.3

  cta_label = "View Course"
  cta_url = "https://bit.ly/android-masterclass"
  cta_icon_pack = "fab"
  cta_icon = "android"

[[item]]
  title = "Recognizing Compositional Actions in Videos with Temporal Ordering"
  content = "As part of my Masters course, I published thesis on Compositional Actions in Videos with Temporal Ordering.<br><br>"
  align = "center"

  overlay_color = "#A663CC"  # Purple for research
  overlay_img = ""
  overlay_filter = 0.3

  cta_label = "View Publications"
  cta_url = "#publications"
  cta_icon_pack = "fas"
  cta_icon = "brain"

[[item]]
  title = "Technical Content Writing"
  content = "Authored 200+ technical articles for leading publications including Draft.dev, CircleCI, Twilio, and Neo4j<br><br>"
  align = "center"

  overlay_color = "#4ECDC4"  # Teal for writing
  overlay_img = ""
  overlay_filter = 0.3

  cta_label = "Read Articles"
  cta_url = "/post/"
  cta_icon_pack = "fas"
  cta_icon = "pen-fancy"

[advanced]
 # Custom CSS to increase space between content and CTA button
 css_style = """
 .hero-slide .hero-content p {
   margin-bottom: 3rem !important;
 }
 .hero-slide .btn {
   margin-top: 2rem !important;
 }
 .wg-slider .carousel-inner .carousel-item .carousel-caption {
   padding-bottom: 4rem !important;
 }
 .wg-slider .btn {
   margin-top: 2rem !important;
 }
 """
 
 # CSS class.
 css_class = ""
+++
