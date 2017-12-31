module.exports = {
  blogPostDir: "sample-posts", // The name of directory that contains your posts.
  siteTitle: "David's Blog", // Site title.
  siteTitleAlt: "A blog site created with GatsbyJS by David Samuel", // Alternative site title for SEO.
  siteLogo: "/logos/logo-512.png", // Logo used for SEO and manifest.
  siteUrl: "https://www.davidsamuel.me", // Domain of your website without pathPrefix.
  pathPrefix: "/", // Prefixes all links. For cases when deployed to example.github.io/gatsby-material-starter/.
  fixedFooter: false, // Whether the footer component is fixed, i.e. always visible
  siteDescription: "A blog about tech, music, and fun stuff.", // Website description used for RSS feeds/meta description tag.
  siteRss: "/rss.xml", // Path to the RSS file.
  siteFBAppID: "170043350414321", // FB Application ID for using app insights
  siteGATrackingID: "UA-111710660-1", // Tracking code ID for google analytics.
  disqusShortname: "davidsamuel", // Disqus shortname.
  postDefaultCategoryID: "Tech", // Default category for posts.
  userName: "majickdave", // Username to display in the author segment.
  userTwitter: "majickdave", // Optionally renders "Follow Me" in the UserInfo segment.
  userLocation: "Los Angeles, California, USA", // User location to display in the author segment.
  userAvatar: "https://www.dropbox.com/s/u7egnn54iznzp61/me_avatar_small.png?raw=1", // User avatar to display in the author segment.
  userDescription:
    "Data Scientist, and technology nerd", // User description to display in the author segment.
  // Links to social profiles/projects you want to display in the author segment/navigation bar.
  userLinks: [
    {
      label: "Medium",
      url: "https://medium.com/@davesamuel",
      iconClassName: "fa fa-medium"
    },
    {
      label: "Twitter",
      url: "https://twitter.com/majickdave",
      iconClassName: "fa fa-twitter"
    },
    {
      label: "Linkedin",
      url: "https://linkedin.com/in/datascienceinsight",
      iconClassName: "fa fa-linkedin"
    }
  ],
  copyright: "Copyright © 2017. David Samuel" // Copyright string for the footer of the website and RSS feed.
};
