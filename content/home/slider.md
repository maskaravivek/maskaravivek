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
  title = "mdedit.ai"
  content = "An AI-powered Markdown editor built for developers and tech writers. 8,000+ writers use it to write, edit, and publish seamlessly.<br><br>"
  align = "center"

  overlay_color = "#2E86AB"
  overlay_filter = 0.4

  cta_label = "Try mdedit.ai"
  cta_url = "https://mdedit.ai"
  cta_icon_pack = "fas"
  cta_icon = "edit"

[[item]]
  title = "Automated i18n for Any App"
  content = "gpt-localize-action is a GitHub Action that keeps your translation files in sync automatically. It detects changes, translates only the diff, and opens a PR. Running across 7+ of my own apps.<br><br>"
  align = "center"

  overlay_color = "#A663CC"
  overlay_filter = 0.4

  cta_label = "View on GitHub"
  cta_url = "https://github.com/mangoappstudio/gpt-localize-action"
  cta_icon_pack = "fab"
  cta_icon = "github"

[[item]]
  title = "Grain Quality Analysis with AI"
  content = "Inweon GRAMS uses computer vision to assess rice, wheat, pulses, and oilseeds with ~99% accuracy. I built the cloud infrastructure, mobile app, and ML labeling platform.<br><br>"
  align = "center"

  overlay_color = "#4C8C2B"
  overlay_filter = 0.4

  cta_label = "Check out GRAMS"
  cta_url = "https://inweon.com"
  cta_icon_pack = "fas"
  cta_icon = "leaf"

[[item]]
  title = "Android Development Masterclass"
  content = "A hands-on course with 60+ lessons and 40+ interactive example apps. Learn to build complex Android apps the right way, available on Educative.<br><br>"
  align = "center"

  overlay_color = "#3DDC84"
  overlay_img = ""
  overlay_filter = 0.3

  cta_label = "View Course"
  cta_url = "https://bit.ly/android-masterclass"
  cta_icon_pack = "fab"
  cta_icon = "android"

[[item]]
  title = "200+ Technical Articles"
  content = "Writing is how I give back to the dev community. I've published articles for Draft.dev, CircleCI, Twilio, Neo4j, and others, covering mobile, cloud, and everything in between.<br><br>"
  align = "center"

  overlay_color = "#4ECDC4"
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
