export default function BlogPage() {
  const blogs = [
    {
      title: "Understanding Alzheimer's: Causes and Symptoms",
      author: "Dr. A. Sharma",
      date: "Feb 12, 2025",
      description:
        "Alzheimer’s disease is a progressive neurological disorder that affects memory and thinking abilities. Learn about the risk factors, early symptoms, and possible treatment options available today.",
      image: "https://via.placeholder.com/400x250", // Replace with actual image
    },
    {
      title: "Early Signs of Alzheimer’s and How to Detect Them",
      author: "Dr. P. Verma",
      date: "Feb 10, 2025",
      description:
        "Early detection of Alzheimer’s can help in managing the disease better. This blog covers common symptoms and early-stage diagnosis techniques that can be helpful.",
      image: "https://via.placeholder.com/400x250",
    },
  ];

  return (
    <div className="min-h-screen bg-[#121212] text-[#E0E0E0] p-6">
      <h1 className="text-3xl font-bold text-[#BB86FC] mb-6">Blog</h1>
      <div className="grid md:grid-cols-2 gap-6">
        {blogs.map((blog, index) => (
          <div key={index} className="bg-[#1E1E1E] p-6 rounded-2xl shadow-lg">
            <img
              src={blog.image}
              alt={blog.title}
              className="w-full h-40 object-cover rounded-lg mb-4"
            />
            <h2 className="text-xl font-semibold text-[#BB86FC]">
              {blog.title}
            </h2>
            <p className="text-sm text-[#BDBDBD]">
              {blog.author} • {blog.date}
            </p>
            <p className="mt-2 text-[#E0E0E0]">{blog.description}</p>
            <button className="mt-4 bg-[#BB86FC] text-[#121212] px-4 py-2 rounded-lg">
              Read More
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
