---
title: "SPARQL session at the Wikimedia Prague Pre-Hackathon"
author: "Vivek Maskara"
date: 2017-05-13T09:47:59.000Z
lastmod: 2021-10-29T21:23:12-07:00

description: ""

subtitle: ""

tags: [Open Source, Wikimedia, SPARQL]
categories: [Open Source]

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2017-05-13_sparql-session-at-the-wikimedia-prague-prehackathon_0.jpg"
 - "/post/img/2017-05-13_sparql-session-at-the-wikimedia-prague-prehackathon_1.png"
 - "/post/img/2017-05-13_sparql-session-at-the-wikimedia-prague-prehackathon_2.png"


aliases:
- "/sparql-session-at-the-wikimedia-prague-pre-hackathon-5e7ded44fb4e"

---

![](/post/img/2017-05-13_sparql-session-at-the-wikimedia-prague-prehackathon_0.jpg#layoutTextWidth)

Am at the Prague pre-hackathon and the guys from Wikipedia UK and Wikipedia Austria gave a session on [SPARQL queries](https://en.wikipedia.org/wiki/SPARQL?oldformat=true). It seems to be a very cool way to get results from Wikidata and visualize it. [Adam Shorland](https://addshore.com/) and [Tobias Schönberg](https://github.com/tobias47n9e) took this session for all the Wikimedia Commons App team in Wikimedia Czech Republic office.

They gave us an introduction to get started with SPARQL queries. SPARQL is an RDF query language, that is, a semantic query language for databases, able to retrieve and manipulate data stored in Resource Description Framework (RDF) format.

Here are a few sample queries that can give you a feeling of how it works.

### Query to see a list of world heritage sites

```
SELECT ?item ?itemLabel ?coord ?image { ?item wdt:P1435 wd:Q9259 . ?item wdt:P17 ?country . ?item wdt:P625 ?coord . ?item wdt:P18 ?image SERVICE wikibase:label {bd:serviceParam wikibase:language "en"} }
```

See the results [here](https://query.wikidata.org/embed.html#SELECT%20%3Fitem%20%3FitemLabel%20%3Fcoord%20%3Fimage%0A%7B%0A%09%3Fitem%20wdt%3AP1435%20wd%3AQ9259%20.%0A%20%20%09%3Fitem%20wdt%3AP17%20%3Fcountry%20.%0A%20%20%09%3Fitem%20wdt%3AP625%20%3Fcoord%20.%0A%20%20%09%3Fitem%20wdt%3AP18%20%3Fimage%0A%20%20%20%20SERVICE%20wikibase%3Alabel%20%20%7Bbd%3AserviceParam%20wikibase%3Alanguage%20%22en%22%7D%0A%7D). Here’s a sample output.

![](/post/img/2017-05-13_sparql-session-at-the-wikimedia-prague-prehackathon_1.png#layoutTextWidth)

Query to show a list of nearby places without any images

The [Android Wikimedia Commons app](https://play.google.com/store/apps/details?id=fr.free.nrw.commons) uses this query to get a list of nearby places which don’t have an image on Wikimedia Commons. Anyone is welcome to [contribute to the Android app](https://github.com/commons-app/apps-android-commons).

```
SELECT
     (SAMPLE(?location) as ?location)
     ?item
     (SAMPLE(COALESCE(?item_label_preferred_language, ?item_label_any_language)) as ?label)
     (SAMPLE(?classId) as ?class)
     (SAMPLE(COALESCE(?class_label_preferred_language, ?class_label_any_language, "?")) as ?class_label)
     (SAMPLE(COALESCE(?icon0, ?icon1)) as ?icon)
     (SAMPLE(COALESCE(?emoji0, ?emoji1)) as ?emoji)
     ?wikipediaArticle
     ?commonsArticle
   WHERE {
     # Around given location...
     SERVICE wikibase:around {
       ?item wdt:P625 ?location.
       bd:serviceParam wikibase:center "Point(${LONG} ${LAT})"^^geo:wktLiteral.
       bd:serviceParam wikibase:radius "${RAD}" . # Radius in kilometers.
     }

     # ... and without an image.
     MINUS {?item wdt:P18 []}

     # Get the label in the preferred language of the user, or any other language if no label is available in that language.
     OPTIONAL {?item rdfs:label ?item_label_preferred_language. FILTER (lang(?item_label_preferred_language) = "${LANG}")}
     OPTIONAL {?item rdfs:label ?item_label_any_language}

     # Get the class label in the preferred language of the user, or any other language if no label is available in that language.
     OPTIONAL {
       ?item p:P31/ps:P31 ?classId.
       OPTIONAL {?classId rdfs:label ?class_label_preferred_language. FILTER (lang(?class_label_preferred_language) = "${LANG}")}
       OPTIONAL {?classId rdfs:label ?class_label_any_language}

       # Get icon
       OPTIONAL { ?classId wdt:P2910 ?icon0. }
       OPTIONAL { ?classId wdt:P279*/wdt:P2910 ?icon1. }
       # Get emoji
       OPTIONAL { ?classId wdt:P487 ?emoji0. }
       OPTIONAL { ?classId wdt:P279*/wdt:P487 ?emoji1. }
       OPTIONAL {
          ?sitelink schema:about ?item .
          ?sitelink schema:inLanguage "en"
       }
       OPTIONAL {
           ?wikipediaArticle   schema:about ?item ;
                               schema:isPartOf <https://en.wikipedia.org/> .
           SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
         }

         OPTIONAL {
           ?commonsArticle   schema:about ?item ;
                               schema:isPartOf <https://commons.wikimedia.org/> .
           SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
         }
     }
   }
   GROUP BY ?item ?wikipediaArticle ?commonsArticle
```

See the results [here](https://query.wikidata.org/embed.html#SELECT%0A%28SAMPLE%28%3Flocation%29%20as%20%3Flocation%29%0A%3Fitem%0A%28SAMPLE%28COALESCE%28%3Fitem_label_preferred_language%2C%20%3Fitem_label_any_language%29%29%20as%20%3Flabel%29%0A%28SAMPLE%28%3FclassId%29%20as%20%3Fclass%29%0A%28SAMPLE%28COALESCE%28%3Fclass_label_preferred_language%2C%20%3Fclass_label_any_language%2C%20%22%3F%22%29%29%20as%20%3Fclass_label%29%0A%28SAMPLE%28COALESCE%28%3Ficon0%2C%20%3Ficon1%29%29%20as%20%3Ficon%29%0A%28SAMPLE%28COALESCE%28%3Femoji0%2C%20%3Femoji1%29%29%20as%20%3Femoji%29%0A%3FwikipediaArticle%0A%3FcommonsArticle%0AWHERE%20%7B%0A%20%20%23%20Around%20given%20location...%0A%20%20SERVICE%20wikibase%3Aaround%20%7B%0A%20%20%20%20%3Fitem%20wdt%3AP625%20%3Flocation.%0A%20%20%20%20bd%3AserviceParam%20wikibase%3Acenter%20%22Point%2877.6435%2012.9585%29%22%5E%5Egeo%3AwktLiteral.%0A%20%20%20%20bd%3AserviceParam%20wikibase%3Aradius%20%222.62%22%20.%20%23%20Radius%20in%20kilometers.%0A%20%20%7D%0A%0A%20%20%23%20...%20and%20without%20an%20image.%0A%20%20MINUS%20%7B%3Fitem%20wdt%3AP18%20%5B%5D%7D%0A%0A%20%20%23%20Get%20the%20label%20in%20the%20preferred%20language%20of%20the%20user%2C%20or%20any%20other%20language%20if%20no%20label%20is%20available%20in%20that%20language.%0A%20%20OPTIONAL%20%7B%3Fitem%20rdfs%3Alabel%20%3Fitem_label_preferred_language.%20FILTER%20%28lang%28%3Fitem_label_preferred_language%29%20%3D%20%22en%22%29%7D%0A%20%20OPTIONAL%20%7B%3Fitem%20rdfs%3Alabel%20%3Fitem_label_any_language%7D%0A%0A%20%20%23%20Get%20the%20class%20label%20in%20the%20preferred%20language%20of%20the%20user%2C%20or%20any%20other%20language%20if%20no%20label%20is%20available%20in%20that%20language.%0A%20%20OPTIONAL%20%7B%0A%20%20%20%20%3Fitem%20p%3AP31%2Fps%3AP31%20%3FclassId.%0A%20%20%20%20OPTIONAL%20%7B%3FclassId%20rdfs%3Alabel%20%3Fclass_label_preferred_language.%20FILTER%20%28lang%28%3Fclass_label_preferred_language%29%20%3D%20%22en%22%29%7D%0A%20%20%20%20OPTIONAL%20%7B%3FclassId%20rdfs%3Alabel%20%3Fclass_label_any_language%7D%0A%0A%20%20%20%20%23%20Get%20icon%0A%20%20%20%20OPTIONAL%20%7B%20%3FclassId%20wdt%3AP2910%20%3Ficon0.%20%7D%0A%20%20%20%20OPTIONAL%20%7B%20%3FclassId%20wdt%3AP279%2a%2Fwdt%3AP2910%20%3Ficon1.%20%7D%0A%20%20%20%20%23%20Get%20emoji%0A%20%20%20%20OPTIONAL%20%7B%20%3FclassId%20wdt%3AP487%20%3Femoji0.%20%7D%0A%20%20%20%20OPTIONAL%20%7B%20%3FclassId%20wdt%3AP279%2a%2Fwdt%3AP487%20%3Femoji1.%20%7D%0A%20%20%20%20OPTIONAL%20%7B%0A%20%20%20%20%20%20%3Fsitelink%20schema%3Aabout%20%3Fitem%20.%0A%20%20%20%20%20%20%3Fsitelink%20schema%3AinLanguage%20%22en%22%0A%20%20%20%20%7D%0A%20%20%20%20OPTIONAL%20%7B%0A%20%20%20%20%20%20%3FwikipediaArticle%20%20%20schema%3Aabout%20%3Fitem%20%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20schema%3AisPartOf%20%3Chttps%3A%2F%2Fen.wikipedia.org%2F%3E%20.%0A%20%20%20%20%20%20SERVICE%20wikibase%3Alabel%20%7B%20bd%3AserviceParam%20wikibase%3Alanguage%20%22en%22%20%7D%0A%20%20%20%20%7D%0A%0A%20%20%20%20OPTIONAL%20%7B%0A%20%20%20%20%20%20%3FcommonsArticle%20%20%20schema%3Aabout%20%3Fitem%20%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20schema%3AisPartOf%20%3Chttps%3A%2F%2Fcommons.wikimedia.org%2F%3E%20.%0A%20%20%20%20%20%20SERVICE%20wikibase%3Alabel%20%7B%20bd%3AserviceParam%20wikibase%3Alanguage%20%22en%22%20%7D%0A%20%20%20%20%7D%0A%20%20%7D%0A%7D%0AGROUP%20BY%20%3Fitem%20%3FwikipediaArticle%20%3FcommonsArticle). Here’s a sample output.

![](/post/img/2017-05-13_sparql-session-at-the-wikimedia-prague-prehackathon_2.png#layoutTextWidth)

Originally published at [www.maskaravivek.com](http://www.maskaravivek.com/blog/general/sparql-session-at-the-wikimedia-prague-pre-hackathon/) on May 13, 2017.

* * *
Written on May 13, 2017 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/sparql-session-at-the-wikimedia-prague-pre-hackathon-5e7ded44fb4e)
