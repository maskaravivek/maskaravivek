---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Recognizing Compositional Actions in Videos with Temporal Ordering"
authors: [admin]
date: 2022-05-30T18:56:27-07:00

# Schedule page publish date (NOT publication's date).
publishDate: 2022-05-30T18:56:27-07:00

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["7"]

# Publication name and optional abbreviated publication name.
publication: "ProQuest"
publication_short: ""

abstract: "
In some scenarios, true temporal ordering is required to identify the actions occurring in a video. Recently a new synthetic dataset named CATER, was introduced containing 3D objects like sphere, cone, cylinder etc. which undergo simple movements such as slide, pick & place etc. The task defined in the dataset is to identify compositional actions with temporal ordering. In this thesis, a rule-based system and a window-based technique are proposed to identify individual actions (atomic) and multiple actions with temporal ordering (composite) on the CATER dataset. The rule-based system proposed here is a heuristic algorithm that evaluates the magnitude and direction of object movement across frames to determine the atomic action temporal windows and uses these windows to predict the composite actions in the videos. The performance of the rule-based system is validated using the frame-level object coordinates provided in the dataset and it outperforms the performance of the baseline models on the CATER dataset. A window-based training technique is proposed for identifying composite actions in the videos. A pre-trained deep neural network (I3D model) is used as a base network for action recognition. During inference, non-overlapping windows are passed through the I3D network to obtain the atomic action predictions and the predictions are passed through a rule-based system to determine the composite actions. The approach outperforms the state-of-the-art composite action recognition models by 13.37% (mAP 66.47% vs. mAP 53.1%)."

# Summary. An optional shortened abstract.
summary: ""

tags: [Thesis, Compositional Actions, Rule Based System, CATER]
categories: [Thesis]
featured: false

# Custom links (optional).
#   Uncomment and edit lines below to show custom links.
# links:
# - name: Follow
#   url: https://twitter.com
#   icon_pack: fab
#   icon: twitter

url_pdf: https://search.lib.asu.edu/permalink/01ASU_INST/fdcm53/cdi_proquest_journals_2670610608
url_code:
url_dataset:
url_poster:
url_project:
url_slides:
url_source:
url_video:

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects: []

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: ""
---
